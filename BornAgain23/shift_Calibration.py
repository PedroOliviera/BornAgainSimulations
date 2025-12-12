#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from datetime import datetime


def shift_image_vertical(image: np.ndarray,
                         shift_pixels: int,
                         fill_value=np.nan) -> np.ndarray:
    """
    Vertically shift a 2D image array.

    Convention:
      - shift_pixels > 0  -> shift image UP (toward smaller row index)
                             empty space appears at the BOTTOM.
      - shift_pixels < 0  -> shift image DOWN
                             empty space appears at the TOP.
      - shift_pixels == 0 -> returns a copy of the original.

    Parameters
    ----------
    image : np.ndarray
        2D array, e.g. GISAXS intensity map.
    shift_pixels : int
        Number of pixels to shift. Positive = up, negative = down.
    fill_value : float
        Value to put into the newly created empty rows
        (e.g. np.nan or 0.0).

    Returns
    -------
    shifted : np.ndarray
        New 2D array with the same shape as `image`.
    """
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    # Ensure we can store NaNs if requested
    if np.isnan(fill_value):
        image = image.astype(float, copy=False)

    nrows, ncols = image.shape
    shifted = np.full_like(image, fill_value)

    if shift_pixels == 0:
        return image.copy()

    # Shift UP: row i -> i - shift_pixels
    if shift_pixels > 0:
        if shift_pixels >= nrows:
            # Entire image shifted out; everything is fill_value
            return shifted
        shifted[0:nrows - shift_pixels, :] = image[shift_pixels:nrows, :]

    # Shift DOWN: row i -> i + |shift_pixels|
    else:
        shift = -shift_pixels
        if shift >= nrows:
            return shifted
        shifted[shift:nrows, :] = image[0:nrows - shift, :]

    return shifted


def main(input_path="4_12deg.npz",
         output_path="4_12deg_shifted.npz",
         shift_pixels=50,
         fill_value=np.nan):
    """
    Load GISAXS image from an .npz file, apply vertical shift,
    and save result to a new .npz file.

    - Uses the 'sim' array inside the npz as the image.
    - Ignores the extent (axes_limits) for the shift operation.
    """
    input_path = Path(input_path)
    data = np.load(input_path, allow_pickle=True)

    image = data["sim"]            # GISAXS image
    # axes_limits = data["axes_limits"]  # we ignore this for the shift

    shifted_image = shift_image_vertical(image, shift_pixels, fill_value)

    # Preserve other metadata if present
    out_kwargs = {}
    for key in data.files:
        if key == "sim":
            continue
        out_kwargs[key] = data[key]

    # Overwrite 'sim' with shifted image
    out_kwargs["sim"] = shifted_image
    out_kwargs["saved_at_shifted"] = str(datetime.now())

    np.savez(output_path, **out_kwargs)
    print(f"Saved shifted image to: {output_path}")


if __name__ == "__main__":
    # Example usage:
    #   python shift_gisaxs_vertical.py
    #
    # You can edit the values below or convert to argparse.
    main(
        input_path="4_12deg.npz",
        output_path="4_12deg_shifted_nan.npz",
        shift_pixels=50,     # shift UP by 50 pixels
        fill_value=np.nan    # or 0.0 to fill with zeros
    )