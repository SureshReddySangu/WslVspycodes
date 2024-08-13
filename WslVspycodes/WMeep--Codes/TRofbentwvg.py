import meep as mp
import numpy as np
import matplotlib.pyplot as plt

resolution = 10  # pixels/um

sx_values = [16, 32]  # different values of sx to iterate over
sy_values = [32, 64]  # different values of sy to iterate over

fcen = 0.15  # pulse center frequency
df = 0.1  # pulse width (in frequency)
nfreq = 100  # number of frequencies at which to compute flux
dpml = 1.0

all_wl = []
all_Rs = []
all_Ts = []

for sx, sy in zip(sx_values, sy_values):
    cell = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(dpml)]
    pad = 4  # padding distance between waveguide and cell edge
    w = 1  # width of waveguide
    wvg_xcen = 0.5 * (sx - w - 2 * pad)  # x center of horiz. wvg
    wvg_ycen = -0.5 * (sy - w - 2 * pad)  # y center of vert. wvg

    geometry = [mp.Block(size=mp.Vector3(mp.inf, w, mp.inf),
                         center=mp.Vector3(0, wvg_ycen, 0),
                         material=mp.Medium(epsilon=12))]

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                         component=mp.Ez,
                         center=mp.Vector3(-0.5 * sx + dpml, wvg_ycen, 0),
                         size=mp.Vector3(0, w, 0))]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    # reflected flux
    refl_fr = mp.FluxRegion(center=mp.Vector3(-0.5 * sx + dpml + 0.5, wvg_ycen, 0), size=mp.Vector3(0, 2 * w, 0))
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)

    # transmitted flux
    tran_fr = mp.FluxRegion(center=mp.Vector3(0.5 * sx - dpml, wvg_ycen, 0), size=mp.Vector3(0, 2 * w, 0))
    tran = sim.add_flux(fcen, df, nfreq, tran_fr)

    pt = mp.Vector3(0.5 * sx - dpml - 0.5, wvg_ycen)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

    # for normalization run, save flux fields data for reflection plane
    straight_refl_data = sim.get_flux_data(refl)

    # save incident power for transmission plane
    straight_tran_flux = mp.get_fluxes(tran)

    sim.reset_meep()

    geometry = [mp.Block(mp.Vector3(sx - pad, w, mp.inf), center=mp.Vector3(-0.5 * pad, wvg_ycen),
                         material=mp.Medium(epsilon=12)),
                mp.Block(mp.Vector3(w, sy - pad, mp.inf), center=mp.Vector3(wvg_xcen, 0.5 * pad),
                         material=mp.Medium(epsilon=12))]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    # reflected flux
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)

    tran_fr = mp.FluxRegion(center=mp.Vector3(wvg_xcen, 0.5 * sy - dpml - 0.5, 0), size=mp.Vector3(2 * w, 0, 0))
    tran = sim.add_flux(fcen, df, nfreq, tran_fr)

    # for normal run, load negated fields to subtract incident from refl. fields
    sim.load_minus_flux_data(refl, straight_refl_data)

    pt = mp.Vector3(wvg_xcen, 0.5 * sy - dpml - 0.5)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

    bend_refl_flux = mp.get_fluxes(refl)
    bend_tran_flux = mp.get_fluxes(tran)

    flux_freqs = mp.get_flux_freqs(refl)

    wl = []
    Rs = []
    Ts = []
    for i in range(nfreq):
        wl = np.append(wl, 1 / flux_freqs[i])
        Rs = np.append(Rs, -bend_refl_flux[i] / straight_tran_flux[i])
        Ts = np.append(Ts, bend_tran_flux[i] / straight_tran_flux[i])

    all_wl.append(wl)
    all_Rs.append(Rs)
    all_Ts.append(Ts)

if mp.am_master():
    plt.figure()
    for i in range(len(sx_values)):
        if sx_values[i] == 32 and sy_values[i] == 64:
            linestyle = '--'  # dotted line
            color = 'k'  # black color
        else:
            linestyle = '-'  # solid line
            color = None  # default color

        plt.plot(all_wl[i], all_Rs[i], linestyle=linestyle, color=color, label=f'Reflectance (sx={sx_values[i]}, sy={sy_values[i]})')
        plt.plot(all_wl[i], all_Ts[i], linestyle=linestyle, color=color, label=f'Transmittance (sx={sx_values[i]}, sy={sy_values[i]})')
        plt.plot(all_wl[i], 1 - all_Rs[i] - all_Ts[i], linestyle=linestyle, color=color, label=f'Loss (sx={sx_values[i]}, sy={sy_values[i]})')
    plt.axis([5.0, 10.0, 0, 1])
    plt.xlabel("wavelength (μm)")
    plt.legend(loc="upper right")
    plt.show()
