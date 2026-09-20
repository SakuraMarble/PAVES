/*
 * quals.c — classify and compile scalar predicates for engine pushdown.
 *
 * A clause is pushdownable when it is an AND/OR/NOT tree over
 * {Var op Const-like}, {Var IN (...)} or bare boolean Vars, where every Var
 * is a cached scalar column of the indexed relation and every non-Var side
 * is a Var-free, non-volatile expression (Const, Param, stable function).
 *
 * Compilation happens at executor Begin time so Param values are available.
 * Pushdown may only produce false positives w.r.t. SQL semantics (NULL
 * handling, see engine_api.h); the executor rechecks all quals on emitted
 * tuples, so pushdown never affects correctness.
 */
#include "paves.h"

#include <math.h>

#include "access/stratnum.h"
#include "catalog/pg_type.h"
#include "executor/executor.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/optimizer.h"
#include "utils/array.h"
#include "utils/lsyscache.h"

typedef struct CompileCtx {
    FvIndex *idx;
    Index varno;
    /* NULL in dry-run (planner) mode: */
    ExprContext *econtext;
    PlanState *ps;
} CompileCtx;

static bool compile_rec(Node *node, CompileCtx *ctx, FvPredNode **out);

static Node *
strip_relabel(Node *node)
{
    while (node && IsA(node, RelabelType))
        node = (Node *) ((RelabelType *) node)->arg;
    return node;
}

/* Var over a cached scalar column of our rel? Returns col slot or -1. */
static int
cached_var_slot(Node *node, CompileCtx *ctx)
{
    Var *var;

    node = strip_relabel(node);
    if (!node || !IsA(node, Var))
        return -1;
    var = (Var *) node;
    if (var->varno != ctx->varno || var->varlevelsup != 0 || var->varattno <= 0)
        return -1;
    return fv_index_col_slot(ctx->idx, var->varattno);
}

static bool
const_side_ok(Node *node)
{
    return node != NULL &&
           !contain_var_clause(node) &&
           !contain_volatile_functions(node);
}

static bool
datum_to_double(Oid typid, Datum d, double *out)
{
    switch (typid)
    {
        case BOOLOID:
            *out = DatumGetBool(d) ? 1.0 : 0.0;
            return true;
        case INT2OID:
            *out = (double) DatumGetInt16(d);
            return true;
        case INT4OID:
            *out = (double) DatumGetInt32(d);
            return true;
        case INT8OID:
        {
            int64 v = DatumGetInt64(d);
            /* beyond 2^52 double conversion may create false negatives */
            if (v > (int64) 1 << 52 || v < -((int64) 1 << 52))
                return false;
            *out = (double) v;
            return true;
        }
        case FLOAT4OID:
            *out = (double) DatumGetFloat4(d);
            return true;
        case FLOAT8OID:
            *out = DatumGetFloat8(d);
            return true;
        default:
            return false;
    }
}

static bool
type_convertible(Oid typid)
{
    return typid == BOOLOID || typid == INT2OID || typid == INT4OID ||
           typid == INT8OID || typid == FLOAT4OID || typid == FLOAT8OID;
}

/* evaluate a Var-free expression at Begin time */
static bool
eval_const_side(Node *node, CompileCtx *ctx, Datum *d, bool *isnull)
{
    if (IsA(node, Const))
    {
        *d = ((Const *) node)->constvalue;
        *isnull = ((Const *) node)->constisnull;
        return true;
    }
    if (ctx->econtext == NULL)
        return false; /* dry-run never gets here for non-Const */
    {
        ExprState *state = ExecInitExpr((Expr *) node, ctx->ps);
        *d = ExecEvalExprSwitchContext(state, ctx->econtext, isnull);
        return true;
    }
}

static FvPredNode *
mk_node(CompileCtx *ctx)
{
    if (ctx->econtext == NULL)
        return NULL; /* dry-run: shape check only */
    return (FvPredNode *) palloc0(sizeof(FvPredNode));
}

/* constant-false leaf: EQ against NaN never matches */
static void
set_always_false(FvPredNode *n)
{
    if (n == NULL)
        return;
    n->kind = FV_NODE_CMP;
    n->col = 0;
    n->op = FV_CMP_EQ;
    n->value = (double) NAN;
}

static int
btree_strategy(Oid opno, bool var_on_left)
{
    List *interps = get_op_btree_interpretation(opno);
    ListCell *lc;
    int strategy = 0;

    foreach(lc, interps)
    {
        OpBtreeInterpretation *interp = (OpBtreeInterpretation *) lfirst(lc);

        strategy = interp->strategy;
        break;
    }
    list_free_deep(interps);
    if (strategy == 0)
        return 0;
    if (!var_on_left)
    {
        /* commute: const OP var  ->  var OP' const */
        switch (strategy)
        {
            case BTLessStrategyNumber: strategy = BTGreaterStrategyNumber; break;
            case BTLessEqualStrategyNumber: strategy = BTGreaterEqualStrategyNumber; break;
            case BTGreaterEqualStrategyNumber: strategy = BTLessEqualStrategyNumber; break;
            case BTGreaterStrategyNumber: strategy = BTLessStrategyNumber; break;
            default: break; /* EQ / NE symmetric */
        }
    }
    return strategy;
}

static int
strategy_to_cmp(int strategy)
{
    switch (strategy)
    {
        case BTLessStrategyNumber: return FV_CMP_LT;
        case BTLessEqualStrategyNumber: return FV_CMP_LE;
        case BTEqualStrategyNumber: return FV_CMP_EQ;
        case BTGreaterEqualStrategyNumber: return FV_CMP_GE;
        case BTGreaterStrategyNumber: return FV_CMP_GT;
        case ROWCOMPARE_NE: return FV_CMP_NE;
        default: return -1;
    }
}

static bool
compile_opexpr(OpExpr *op, CompileCtx *ctx, FvPredNode **out)
{
    Node *left, *right, *varside, *constside;
    bool var_on_left;
    int slot, strategy, cmp;
    Datum d;
    bool isnull;
    double v;

    if (list_length(op->args) != 2)
        return false;
    left = (Node *) linitial(op->args);
    right = (Node *) lsecond(op->args);

    if (cached_var_slot(left, ctx) >= 0 && const_side_ok(right))
    {
        varside = left;
        constside = right;
        var_on_left = true;
    }
    else if (cached_var_slot(right, ctx) >= 0 && const_side_ok(left))
    {
        varside = right;
        constside = left;
        var_on_left = false;
    }
    else
        return false;

    slot = cached_var_slot(varside, ctx);
    if (!type_convertible(exprType(constside)))
        return false;
    strategy = btree_strategy(op->opno, var_on_left);
    cmp = strategy_to_cmp(strategy);
    if (cmp < 0)
        return false;

    *out = mk_node(ctx);
    if (ctx->econtext == NULL)
        return true; /* dry-run: shape OK */

    if (!eval_const_side(constside, ctx, &d, &isnull))
        return false;
    if (isnull || !datum_to_double(exprType(constside), d, &v))
    {
        /* NULL comparison (or unconvertible int8) never matches */
        set_always_false(*out);
        return true;
    }
    (*out)->kind = FV_NODE_CMP;
    (*out)->col = slot;
    (*out)->op = cmp;
    (*out)->value = v;
    return true;
}

static bool
compile_saop(ScalarArrayOpExpr *sa, CompileCtx *ctx, FvPredNode **out)
{
    Node *left, *right;
    int slot, strategy;
    Datum d;
    bool isnull;
    Oid elemtype;

    if (!sa->useOr || list_length(sa->args) != 2)
        return false;
    left = (Node *) linitial(sa->args);
    right = (Node *) lsecond(sa->args);
    slot = cached_var_slot(left, ctx);
    if (slot < 0 || !const_side_ok(right))
        return false;
    strategy = btree_strategy(sa->opno, true);
    if (strategy_to_cmp(strategy) != FV_CMP_EQ)
        return false;
    elemtype = get_element_type(exprType(right));
    if (!type_convertible(elemtype))
        return false;

    *out = mk_node(ctx);
    if (ctx->econtext == NULL)
        return true;

    if (!eval_const_side(right, ctx, &d, &isnull))
        return false;
    if (isnull)
    {
        set_always_false(*out);
        return true;
    }
    {
        ArrayType *arr = DatumGetArrayTypeP(d);
        Datum *elems;
        bool *nulls;
        int nelems, i, nv = 0;
        int16 typlen;
        bool typbyval;
        char typalign;
        double *values;

        get_typlenbyvalalign(elemtype, &typlen, &typbyval, &typalign);
        deconstruct_array(arr, elemtype, typlen, typbyval, typalign,
                          &elems, &nulls, &nelems);
        values = (double *) palloc(sizeof(double) * Max(nelems, 1));
        for (i = 0; i < nelems; i++)
        {
            if (nulls[i])
                continue; /* NULL element can never equal a stored value */
            if (!datum_to_double(elemtype, elems[i], &values[nv]))
                return false;
            nv++;
        }
        if (nv == 0)
        {
            set_always_false(*out);
            return true;
        }
        (*out)->kind = FV_NODE_IN;
        (*out)->col = slot;
        (*out)->nvalues = nv;
        (*out)->values = values;
    }
    return true;
}

static bool
compile_rec(Node *node, CompileCtx *ctx, FvPredNode **out)
{
    *out = NULL;
    if (node == NULL)
        return false;
    if (IsA(node, RestrictInfo))
        node = (Node *) ((RestrictInfo *) node)->clause;

    switch (nodeTag(node))
    {
        case T_OpExpr:
            return compile_opexpr((OpExpr *) node, ctx, out);
        case T_ScalarArrayOpExpr:
            return compile_saop((ScalarArrayOpExpr *) node, ctx, out);
        case T_BoolExpr:
        {
            BoolExpr *be = (BoolExpr *) node;
            int nch = list_length(be->args);
            ListCell *lc;
            FvPredNode **children = NULL;
            int i = 0;

            if (nch < 1)
                return false;
            if (ctx->econtext)
                children = (FvPredNode **) palloc(sizeof(FvPredNode *) * nch);
            foreach(lc, be->args)
            {
                FvPredNode *child;

                if (!compile_rec((Node *) lfirst(lc), ctx, &child))
                    return false;
                if (children)
                    children[i++] = child;
            }
            *out = mk_node(ctx);
            if (ctx->econtext == NULL)
                return true;
            (*out)->kind = be->boolop == AND_EXPR ? FV_NODE_AND :
                           be->boolop == OR_EXPR ? FV_NODE_OR : FV_NODE_NOT;
            (*out)->nchildren = nch;
            (*out)->children = (const FvPredNode *const *) children;
            return true;
        }
        case T_Var:
        case T_RelabelType:
        {
            /* bare boolean column reference */
            Node *v = strip_relabel(node);
            int slot = cached_var_slot(v, ctx);

            if (slot < 0 || exprType(v) != BOOLOID)
                return false;
            *out = mk_node(ctx);
            if (ctx->econtext == NULL)
                return true;
            (*out)->kind = FV_NODE_CMP;
            (*out)->col = slot;
            (*out)->op = FV_CMP_EQ;
            (*out)->value = 1.0;
            return true;
        }
        default:
            return false;
    }
}

bool
paves_clause_pushdownable(Node *clause, FvIndex *idx, Index varno)
{
    CompileCtx ctx = {idx, varno, NULL, NULL};
    FvPredNode *dummy;

    return compile_rec(clause, &ctx, &dummy);
}

FvPredNode *
paves_compile_clause(Node *clause, FvIndex *idx, Index varno,
                       ExprContext *econtext, PlanState *ps)
{
    CompileCtx ctx = {idx, varno, econtext, ps};
    FvPredNode *out;

    if (!compile_rec(clause, &ctx, &out))
        return NULL;
    return out;
}
