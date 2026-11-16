# Sample data

## `uci_student_dropout.csv`

The original, unmodified file from the UCI Machine Learning Repository:

> Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout
> and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

License: CC BY 4.0. 4,424 students of a Portuguese higher-education institution; `Target` is
`Dropout`, `Graduate` or `Enrolled` (still studying, so the outcome is unknown and those students
get scored). The file is kept exactly as published (semicolon-separated, UTF-8 with BOM, a tab in
one header) because it is a realistic test of the upload parser.

SHA-256: `3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae`

## `college_template.csv`

Header row only: a suggested layout for a college's own records. Fill one row per student and
leave `result` blank for students whose result is not out yet.
