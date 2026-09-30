"""Compatible, explicitly coordinate-matched wing contrasts."""

from mce.config.settings import OptionsContextSettings
from mce.features.options_context.contracts import ATMIVPoint, SurfaceIVPoint
from mce.features.options_context.models import SkewContext
from mce.models.enums import DataQuality


def _wing(
    points: tuple[SurfaceIVPoint, ...],
    side: str,
    atm: ATMIVPoint,
    settings: OptionsContextSettings,
) -> SurfaceIVPoint | None:
    expected = -settings.wing_abs_coordinate if side == "put" else settings.wing_abs_coordinate
    compatible = (
        point
        for point in points
        if point.side == side
        and point.coordinate_system == settings.skew_coordinate
        and point.expiration_ns == atm.expiration_ns
        and abs(point.coordinate - expected) <= settings.coordinate_tolerance
        and point.quality not in (DataQuality.POOR, DataQuality.UNUSABLE)
    )
    return min(
        compatible,
        key=lambda point: (abs(point.dte_days - atm.dte_days), abs(point.coordinate - expected)),
        default=None,
    )


def skew_context(
    atm: ATMIVPoint,
    atm_points: tuple[ATMIVPoint, ...],
    surface: tuple[SurfaceIVPoint, ...],
    settings: OptionsContextSettings,
) -> SkewContext:
    put = _wing(surface, "put", atm, settings)
    call = _wing(surface, "call", atm, settings)
    put_iv = put.implied_volatility if put else None
    call_iv = call.implied_volatility if call else None
    same_tenor = tuple(point for point in atm_points if point.expiration_ns == atm.expiration_ns)
    atm_call = next(
        (point.implied_volatility for point in same_tenor if point.combination == "call"), None
    )
    atm_put = next(
        (point.implied_volatility for point in same_tenor if point.combination == "put"), None
    )
    return SkewContext(
        coordinate_system=settings.skew_coordinate,
        put_coordinate=put.coordinate if put else None,
        call_coordinate=call.coordinate if call else None,
        put_iv=put_iv,
        call_iv=call_iv,
        downside=put_iv - atm.implied_volatility if put_iv is not None else None,
        upside=call_iv - atm.implied_volatility if call_iv is not None else None,
        risk_reversal_call_minus_put=call_iv - put_iv
        if call_iv is not None and put_iv is not None
        else None,
        butterfly=put_iv / 2 + call_iv / 2 - atm.implied_volatility
        if put_iv is not None and call_iv is not None
        else None,
        atm_call_minus_put=atm_call - atm_put
        if atm_call is not None and atm_put is not None
        else None,
    )
