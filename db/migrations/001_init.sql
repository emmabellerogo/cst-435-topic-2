-- 001_init.sql
-- Schema for "Income Insight": MLP classifier on the real UCI Adult Income dataset.
-- Apply in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- (Not applied automatically; review first.)

-- ---------------------------------------------------------------------------
-- adult_income: one row per UCI Adult example (loaded once by db/load.py).
-- The raw '?' markers in the source file are stored as NULL (workclass,
-- occupation, native_country) so the sklearn imputer sees real missing values.
-- ---------------------------------------------------------------------------
create table if not exists adult_income (
    id              bigint generated always as identity primary key,
    age             smallint    not null check (age between 0 and 120),
    workclass       text,
    fnlwgt          integer     not null,
    education       text        not null,
    education_num   smallint    not null,
    marital_status  text        not null,
    occupation      text,
    relationship    text        not null,
    race            text        not null,
    sex             text        not null check (sex in ('Male', 'Female')),  -- protected attribute
    capital_gain    integer     not null default 0,
    capital_loss    integer     not null default 0,
    hours_per_week  smallint    not null,
    native_country  text,
    income          text        not null check (income in ('<=50K', '>50K')),
    -- Numeric target derived from income (1 = '>50K'), so SQL never string-matches.
    income_label    smallint generated always as (case when income = '>50K' then 1 else 0 end) stored,
    -- One fixed project-wide split, assigned by db/load.py with a fixed seed, so
    -- every controlled experiment sees identical train/val/test rows and the
    -- fairness audit can restrict itself to held-out ('test') rows.
    split           text        check (split in ('train', 'val', 'test')),
    created_at      timestamptz not null default now()
);

create index if not exists idx_adult_income_split      on adult_income (split);
create index if not exists idx_adult_income_sex_label  on adult_income (sex, income_label);

-- ---------------------------------------------------------------------------
-- runs: one row per training run (one controlled experiment).
-- Searchable hyperparameters are explicit columns; detailed metric dictionaries
-- are JSONB. `config` keeps the full YAML/CLI config for exact reproduction.
-- ---------------------------------------------------------------------------
create table if not exists runs (
    id                 bigint generated always as identity primary key,
    name               text        not null,                  -- e.g. 'baseline', 'wide', 'deep-dropout'
    architecture       text        not null default 'mlp',
    hidden_sizes       integer[]   not null check (cardinality(hidden_sizes) >= 2),
    activation         text        not null,
    dropout            double precision not null check (dropout >= 0 and dropout < 1),
    learning_rate      double precision not null check (learning_rate > 0),
    weight_decay       double precision not null default 0 check (weight_decay >= 0),
    epochs             integer     not null check (epochs > 0),  -- budget requested
    best_epoch         integer,                                  -- epoch of the kept checkpoint
    batch_size         integer     not null check (batch_size > 0),
    seed               integer     not null,
    config             jsonb       not null default '{}',

    -- Headline metrics (explicit so they can be sorted/filtered in SQL).
    train_loss         double precision,
    val_loss           double precision,
    val_accuracy       double precision,
    val_roc_auc        double precision,
    test_accuracy      double precision,
    test_precision     double precision,
    test_recall        double precision,
    test_f1            double precision,
    test_roc_auc       double precision,

    -- Calibration (fit on validation only; test used for reporting only).
    calibration_method text,                                     -- 'platt' | 'isotonic' | 'temperature' | null
    ece_before         double precision,                         -- expected calibration error, uncalibrated
    ece_after          double precision,                         -- ... after calibration
    brier_before       double precision,
    brier_after        double precision,
    calibration_bins   jsonb,                                    -- reliability-diagram data

    -- Detailed metric dictionaries.
    train_metrics      jsonb       not null default '{}',
    val_metrics        jsonb       not null default '{}',
    test_metrics       jsonb       not null default '{}',
    confusion_matrix   jsonb,                                    -- [[tn, fp], [fn, tp]] on test
    per_class_metrics  jsonb,                                    -- precision/recall/f1/support per class
    permutation_importance jsonb,

    -- Provenance.
    checkpoint_path    text,                                     -- e.g. models/best.pt
    preprocessor_path  text,
    is_best            boolean     not null default false,       -- the selected run served by the API
    created_at         timestamptz not null default now()
);

-- At most one run may be flagged as the selected/best run.
create unique index if not exists uq_runs_single_best on runs (is_best) where is_best;
create index if not exists idx_runs_created_at on runs (created_at desc);
create index if not exists idx_runs_val_auc    on runs (val_roc_auc desc);

-- ---------------------------------------------------------------------------
-- predictions: one row per served prediction (the prediction / audit log).
-- request_hash is a hash of the input payload for traceability ONLY; it does
-- not and cannot carry a true label. Ground truth comes solely from the
-- nullable adult_income_id link, which is set only when the prediction was made
-- for a labeled Adult row (audit runs against the held-out split). Arbitrary
-- user predictions leave it NULL and are excluded from FPR/FNR.
-- ---------------------------------------------------------------------------
create table if not exists predictions (
    id               bigint generated always as identity primary key,
    request_hash     text        not null,
    predicted_label  smallint    not null check (predicted_label in (0, 1)),
    predicted_proba  double precision not null check (predicted_proba between 0 and 1),
    served_by_run_id bigint      not null references runs (id) on delete cascade,
    adult_income_id  bigint      references adult_income (id) on delete set null,
    created_at       timestamptz not null default now()
);

create index if not exists idx_predictions_run_id      on predictions (served_by_run_id);
create index if not exists idx_predictions_request_hash on predictions (request_hash);
create index if not exists idx_predictions_created_at   on predictions (created_at desc);
create index if not exists idx_predictions_adult_id     on predictions (adult_income_id)
    where adult_income_id is not null;
-- Audit is reproducible: at most one prediction per (run, labeled source row).
create unique index if not exists uq_predictions_run_adult
    on predictions (served_by_run_id, adult_income_id)
    where adult_income_id is not null;

-- ---------------------------------------------------------------------------
-- v_fairness_audit: FPR / FNR by protected attribute (sex), per run.
-- Only labeled, held-out ('test') rows participate. Aggregates only -- no
-- individual records -- so it is safe to expose read-only to the UI.
-- (Runs with owner privileges, so anon needs no access to the base tables.)
-- ---------------------------------------------------------------------------
create or replace view v_fairness_audit as
select
    p.served_by_run_id                                              as run_id,
    a.sex                                                           as group_value,
    count(*)                                                        as n,
    count(*) filter (where a.income_label = 1 and p.predicted_label = 1) as tp,
    count(*) filter (where a.income_label = 0 and p.predicted_label = 1) as fp,
    count(*) filter (where a.income_label = 0 and p.predicted_label = 0) as tn,
    count(*) filter (where a.income_label = 1 and p.predicted_label = 0) as fn,
    (count(*) filter (where a.income_label = 0 and p.predicted_label = 1))::double precision
        / nullif(count(*) filter (where a.income_label = 0), 0)     as fpr,
    (count(*) filter (where a.income_label = 1 and p.predicted_label = 0))::double precision
        / nullif(count(*) filter (where a.income_label = 1), 0)     as fnr
from predictions p
join adult_income a on a.id = p.adult_income_id
where a.split = 'test'
group by p.served_by_run_id, a.sex;

-- ---------------------------------------------------------------------------
-- Row Level Security.
--   * FastAPI uses the SERVICE-ROLE key, which bypasses RLS: all writes and all
--     reads of adult_income / predictions stay server-side.
--   * anon (Streamlit, public key) gets read-only access to `runs` (metrics
--     only, no sensitive data) and to the aggregate `v_fairness_audit` view.
--   * RLS is enabled on every table; no anon/authenticated write policy exists,
--     and table privileges are revoked as defense in depth.
-- ---------------------------------------------------------------------------
alter table adult_income enable row level security;
alter table runs         enable row level security;
alter table predictions  enable row level security;

revoke all on adult_income, runs, predictions from anon, authenticated;
revoke all on v_fairness_audit               from anon, authenticated;

grant select on runs             to anon;
grant select on v_fairness_audit to anon;

drop policy if exists "anon can read runs" on runs;
create policy "anon can read runs"
    on runs for select
    to anon
    using (true);
