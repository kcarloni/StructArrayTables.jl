
"""
    StructArray{T, N, SA} <: AbstractArray{T, N}

Thin wrapper around `StructArrays.StructArray` that adds:
- Virtual property access via `getproperty` (e.g. `sa.x` for computed properties)
- `propertynames` that reflects element-level properties
- Pretty-printing borrowed from TypedTables
"""
struct StructArray{T, N, SA <: _StructArray{T, N}} <: AbstractArray{T, N}
    data::SA
end

StructArray(; kwargs...) = StructArray(_StructArray(; kwargs...))
StructArray{T}(args...; kwargs...) where {T} = StructArray(_StructArray{T}(args...; kwargs...))

# From a NamedTuple of column arrays
StructArray(nt::NamedTuple) = StructArray(_StructArray(nt))

# From an arbitrary array/iterable (e.g. Vector{OBB{Float64}})
StructArray(v::AbstractArray) = StructArray(_StructArray(v))

# ── TypedTables-style constructors ───────────────────────────────────────

"""
    _columns(t::StructArray) -> NamedTuple

Extract the underlying column storage as a NamedTuple.
"""
_columns(t::StructArray) = StructArrays.components(getfield(t, :data))
_columns(nt::NamedTuple) = nt

"""
    _removenothings(nt::NamedTuple) -> NamedTuple

Filter out entries whose value is `nothing`, used for column removal.
"""
_removenothings(nt::NamedTuple) = _removenothings(keys(nt), values(nt))
_removenothings(::Tuple{}, ::Tuple{}) = NamedTuple()
function _removenothings(names::Tuple, vals::Tuple)
    rest = _removenothings(Base.tail(names), Base.tail(vals))
    first(vals) === nothing ? rest : merge(NamedTuple{(first(names),)}((first(vals),)), rest)
end

"""
    StructArray(sources...; kwargs...)

TypedTables-style merging constructor.

    StructArray(a = [1,2], b = [3.0, 4.0])              # keyword columns
    StructArray(existing)                                 # copy/rewrap
    StructArray(existing; c = [true, false])              # extend with new columns
    StructArray(existing; b = nothing)                    # remove a column
    StructArray(t1, t2)                                   # merge columns from multiple sources
"""
function StructArray(sources::Union{StructArray, NamedTuple}...; kwargs...)
    merged = foldl(merge, map(_columns, sources))
    all = merge(merged, values(kwargs))
    StructArray(_removenothings(all))
end

# ── AbstractArray interface ──────────────────────────────────────────────

Base.size(t::StructArray) = size(getfield(t, :data))
Base.axes(t::StructArray) = axes(getfield(t, :data))

Base.IndexStyle(::Type{<:StructArray{T, N, SA}}) where {T, N, SA} = Base.IndexStyle(SA)

Base.@propagate_inbounds Base.getindex(t::StructArray, i::Int) =
    getfield(t, :data)[i]
Base.@propagate_inbounds Base.getindex(t::StructArray{T, N}, I::Vararg{Int, N}) where {T, N} =
    getfield(t, :data)[I...]
Base.@propagate_inbounds Base.setindex!(t::StructArray, val, i::Int) =
    (getfield(t, :data)[i] = val)
Base.@propagate_inbounds Base.setindex!(t::StructArray{T, N}, val, I::Vararg{Int, N}) where {T, N} =
    (getfield(t, :data)[I...] = val)

function Base.similar(t::StructArray, ::Type{S}, dims::Dims) where {S}
    StructArray(similar(getfield(t, :data), S, dims))
end

Base.push!(t::StructArray, val) = (push!(getfield(t, :data), val); t)
Base.append!(t::StructArray, vals) = (append!(getfield(t, :data), vals); t)

Base.copy(t::StructArray) = StructArray(copy(getfield(t, :data)))

# ── Show (borrowed from TypedTables) ────────────────────────────────────

TypedTables.columnnames(t::StructArray) = propertynames(t)

function Base.show(io::IO, ::MIME"text/plain", t::StructArray)
    TypedTables.showtable(io, t)
end

# ── Virtual property access ─────────────────────────────────────────────

"""
    _collect_leaves(x)

If `x` is a `StructArray` whose element type is a Unitful `Quantity`, materialize
it into a plain `Vector`.  Otherwise return `x` unchanged.

StructArrays' broadcast style automatically decomposes composite types into
separate columns.  This is useful for structs (e.g. `Coordinate` -> `point` +
`coordinate_system`), but undesirable for `Quantity`, where it would split the
numeric value from its unit.  Calling `collect` collapses the StructArray back
into a `Vector{Quantity{...}}`, preserving the value-unit pairing.
"""
_collect_leaves(x) = x isa _StructArray && eltype(x) <: Quantity ? collect(x) : x

"""
    _maybe_wrap(x)

Wrap the result of a property access so that chained dot-access keeps working.
- Raw `StructArrays.StructArray` → wrap directly.
- `AbstractArray` whose element type is a composite struct (has fields) → convert
  to `StructArray` so further `.field` access works.
- Everything else (scalars, arrays of primitives/Quantity) → pass through.
"""
_maybe_wrap(x::_StructArray) = StructArray(x)
function _maybe_wrap(x::AbstractArray)
    T = eltype(x)
    isstructtype(T) && fieldcount(T) > 0 && !(T <: Number) && return StructArray(_StructArray(x))
    return x
end
_maybe_wrap(x) = x

"""
    Base.getproperty(t::StructArray, s::Symbol)

Extended property access for `StructArray` that supports virtual properties
defined via custom `Base.getproperty` methods on the element type.

Lookup order:
1. **Real fields** (`hasfield`): delegates to `StructArrays.component`, which
   returns the underlying column array directly (no element reconstruction).
2. **Virtual properties** (e.g. `Coordinate`'s `.x`, `.y`, `.z`): broadcasts
   `getproperty` over each element, reconstructing elements one-by-one.
   The result is passed through [`_collect_leaves`](@ref) to prevent Unitful
   `Quantity` values from being decomposed into separate `val`/`unit` columns.
"""
function Base.getproperty(t::StructArray, s::Symbol)
    sa = getfield(t, :data)
    T = eltype(sa)
    if hasfield(T, s)
        return _maybe_wrap(StructArrays.component(sa, s))
    else
        return _maybe_wrap(_collect_leaves(getproperty.(sa, s)))
    end
end

"""
    Base.propertynames(t::StructArray)

Returns property names from the element type rather than the wrapper's own fields.
Falls back to the StructArray component names if the array is empty.
"""
function Base.propertynames(t::StructArray)
    sa = getfield(t, :data)
    base = propertynames(StructArrays.components(sa))
    isempty(sa) && return base
    return propertynames(first(sa))
end
