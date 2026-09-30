-- DEC-52: the p-values of SPEC-E2-09 adjusted for the multiple comparisons with Holm's step-down method,
-- over the comparisons of one scheme. p_value stays the unadjusted value.
ALTER TABLE significance ADD COLUMN p_holm REAL;
