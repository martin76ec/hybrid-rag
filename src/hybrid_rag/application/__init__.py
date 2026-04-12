"""Application layer — use-case orchestration.

Application services depend only on the domain layer (entities, value objects,
ports).  They orchestrate domain operations but contain no infrastructure code.
"""