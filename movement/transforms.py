"""Transform and add unit attributes to xarray.DataArray datasets."""

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from movement.validators.arrays import validate_dims_coords


def scale(
    data: xr.DataArray,
    factor: ArrayLike | float = 1.0,
    space_unit: str | None = None,
) -> xr.DataArray:
    """Scale data by a given factor with an optional unit.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to be scaled.
    factor : ArrayLike or float
        The scaling factor to apply to the data. If factor is a scalar (a
        single float), the data array is uniformly scaled by the same factor.
        If factor is an object that can be converted to a 1D numpy array (e.g.
        a list of floats), the length of the resulting array must match the
        length of data array's space dimension along which it will be
        broadcasted.
    space_unit : str or None
        The unit of the scaled data stored as a property in
        xarray.DataArray.attrs['space_unit']. In case of the default (``None``)
        the ``space_unit`` attribute is dropped.

    Returns
    -------
    xarray.DataArray
        The scaled data array with updated ``scale_factor`` and ``space_unit``
        attributes.

    Notes
    -----
    When scale is used multiple times on the same xarray.DataArray,
    the ``space_unit`` attribute is overwritten each time or is dropped
    if ``None`` is passed by default or explicitly.

    This scaling factor is stored as the output array's ``scale_factor``
    attribute as a 1D array of the same length as the ``space``
    dimension. If the input data already has a ``scale_factor``
    attribute, the new scale factor is multiplied with the existing one.

    """
    space_len = data.sizes["space"]
    # Validate dimension names
    if space_len == 2:
        validate_dims_coords(data, {"space": ["x", "y"]})
    else:
        validate_dims_coords(data, {"space": ["x", "y", "z"]})

    # Convert `factor` to a 1D array of length `space_len`
    if np.isscalar(factor):
        factor_1d = np.ones(space_len) * factor
    else:
        factor_1d = np.array(factor).squeeze()
        if factor_1d.ndim != 1:
            raise ValueError(
                "Factor must be an object that can be converted to a 1D numpy"
                f" array, got {factor_1d.ndim}D"
            )
        if factor_1d.shape[0] != space_len:
            raise ValueError(
                f"Factor shape {factor_1d.shape} does not match the shape "
                f"of the space dimension {data.space.values.shape}"
            )

    # Reshape factor for broadcasting
    factor_dims = [1] * data.ndim
    factor_dims[data.get_axis_num("space")] = space_len
    factor_broadcast = factor_1d.reshape(factor_dims)

    # Scale the data and update attributes
    scaled_data = data * factor_broadcast
    _update_attrs_upon_scale(scaled_data, space_len, factor_1d, space_unit)
    return scaled_data


def _update_attrs_upon_scale(
    data: xr.DataArray,
    space_len: int,
    factor_1d: np.ndarray,
    space_unit: str | None,
) -> None:
    """Update the 'scale_factor' and 'space_unit' attributes in-place.

    This function modifies the attributes of the provided xarray.DataArray
    in-place. It updates the ``scale_factor`` attribute by multiplying it with
    the provided ``factor_1d``. If the ``space_unit`` is provided,
    it updates or adds the 'space_unit' attribute.
    If space_unit is None, it removes the 'space_unit' attribute.
    """
    # Update scale_factor
    if "scale_factor" in data.attrs:
        existing = np.array(data.attrs["scale_factor"], dtype=float).squeeze()

        # We assume if it's stored, it is 1D array of same length as space
        # dimension; if not, raise an error
        if existing.ndim != 1:
            raise ValueError(
                "Expected existing 'scale_factor' to be 1D, found "
                f"{existing.ndim}D instead."
            )
        if existing.shape[0] != space_len:
            raise ValueError(
                "Existing scale_factor length does not match current "
                f"'space' dimension ({existing.shape[0]} != {space_len})."
            )
        new_scale_factor = existing * factor_1d
    else:
        new_scale_factor = factor_1d

    data.attrs["scale_factor"] = new_scale_factor

    # Update space_unit
    if space_unit is not None:
        data.attrs["space_unit"] = space_unit
    else:
        data.attrs.pop("space_unit", None)
