-- runs_comparison.sql
-- Read-only: generates the README's controlled-configuration comparison table
-- from the runs table. Every column below exists in db/migrations/001_init.sql;
-- early_stopping_patience is read from the persisted config JSONB.
-- Run it in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- It contains no INSERT/UPDATE/DELETE/DDL and changes nothing.
--
-- The ranking reproduces the selection rule fixed before training:
-- lowest validation BCE, ties broken by higher validation ROC-AUC, then name.
-- Test columns are NULL for every run except the selected one, by design.
select
    row_number() over (
        order by r.val_loss asc, r.val_roc_auc desc, r.name asc
    )                                                                   as rank,
    r.id,
    r.name,
    r.hidden_sizes,
    r.activation,
    r.dropout,
    r.seed,
    r.epochs                                                            as epoch_budget,
    (r.config ->> 'early_stopping_patience')::int                       as patience,
    r.batch_size,
    r.learning_rate,
    r.weight_decay,
    r.best_epoch,
    round(r.val_loss::numeric, 4)                                       as val_loss,
    round(r.val_accuracy::numeric, 4)                                   as val_accuracy,
    round(((r.val_metrics ->> 'f1')::double precision)::numeric, 4)     as val_f1,
    round(r.val_roc_auc::numeric, 4)                                    as val_roc_auc,
    round(r.test_accuracy::numeric, 4)                                  as test_accuracy,
    round(r.test_roc_auc::numeric, 4)                                   as test_roc_auc,
    r.is_best
from runs r
where r.name in ('baseline', 'gelu', 'deep')
order by rank;
