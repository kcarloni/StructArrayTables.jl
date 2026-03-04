
# StructArrayTables.jl

An unholy merger of StructArrays.jl and TypedTables.jl. Except actually it's very convenient. Re-exports `StructArray`, with:
- a modified `getproperty` that returns StructArray objects
- a modified `show` borrowed from `TypedTables.jl`
- `TypedTables`-style constructors