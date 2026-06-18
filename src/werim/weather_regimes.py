from spy4cast import add_cyclic_point_to_data
from ._common import *
from .preprocess import Preprocess
from ._log import *
from ._functions import *

class WeatherRegimes(sp_pr._Procedure):
    cor: FArray
    cor_sig: FArray
    reg: FArray
    reg_sig: FArray
    pvalue: FArray
    PCs: FArray
    fvar: FArray
    alpha: float

    r: FArray
    sigma: FArray
    q: FArray
    _psi: Op[FArray]

    VAR_NAMES = (
        'r', 'q', 'sigma',
        'cor', 'cor_sig',
        'reg', 'reg_sig',
        'pvalue', 'PCs',
        'fvar', 'alpha', 'nm',
    )

    @property
    def var_names(self) -> Tuple[str, ...]:
        return self.VAR_NAMES

    def __init__(
        self,
        ds: Preprocess,
        nm: int,
        alpha: float,
        sig: Literal["test-t", "monte-carlo"] = "test-t",
        montecarlo_iterations: Op[int] = None,
        num_svdvals: Op[int] = None,
    ) -> None:
        self._ds = ds

        log_info(f"""Calculating Weather Regimes
    Shape:  {self._ds.shape} 
    Region: {region2str(self._ds.region)}""", )
        here = time_from_here()

        land_data = self._ds.land_data
        x = land_data.not_land_values
        ns, nt = x.shape
        c = np.dot(x, np.transpose(x))

        r, d, q = cast(Tuple[FArray, FArray, FArray], sparse_linalg.svds(c, k=nm, which="LM"))  # Which LM = Large magnitude
        assert r is not None and q is not None

        # Modes are reversed so we reverse them in r, d and q
        self.eof = r[:, ::-1]  # ns x nm
        self.singular_values = d[::-1]  # nm
        self.eof2 = q[::-1, :]  # nm x ns

        if num_svdvals is None:
            svdvals = scipy_linalg.svdvals(c)
            sum_covfrac = np.sum(svdvals ** 2)  # type: ignore
        elif num_svdvals <= nm:
            sum_covfrac = np.sum(self.singular_values[:num_svdvals] ** 2)
        else:
            svdvals = sparse_linalg.svds(c, k=num_svdvals, which='LM', return_singular_vectors=False)
            sum_covfrac = np.sum(svdvals ** 2)  # type: ignore
        self.fvar = self.singular_values ** 2 / sum_covfrac

        PC = np.dot(self.eof.T, x)
        self.PCs = (PC - np.mean(PC)) / np.std(PC)

        self.cor = np.zeros([ns, nm], dtype=np.float32)
        self.cor_sig = np.zeros([ns, nm], dtype=np.float32)
        self.reg = np.zeros([ns, nm], dtype=np.float32)
        self.reg_sig = np.zeros([ns, nm], dtype=np.float32)
        self.pvalue = np.zeros([ns, nm], dtype=np.float32)
        self.alpha = alpha
        self.nm = nm

        for i in range(nm):
            (
                self.cor[:, i],
                self.pvalue[:, i],
                self.cor_sig[:, i],
                self.reg[:, i],
                self.reg_sig[:, i]
            ) = sp_sp.index_regression(land_data, self.PCs[i, :], alpha, sig, montecarlo_iterations)

        log_debug(f'    Took: {time_to_here(here):.03f} seconds', prefix="", info=False)

    def plot(
        self,
        *,
        save_fig: bool = False,
        show_plot: bool = False,
        halt_program: bool = False,
        folder: Op[str] = None,
        name: Op[str] = None,
        figsize: Op[Tuple[float, float]] = None,
        cmap: str = 'bwr',
        nm: Op[int] = None,
        variable: Literal["cor", "reg"] = "cor",
        signs: Op[Sequence[bool]] = None,
        levels: Op[
            Union[npt.NDArray[np.float32], Sequence[float], bool]
        ] = None,
        ticks: Op[Union[npt.NDArray[np.float32], Sequence[float]]] = None,
        plot_type: Op[Literal["contour", "pcolor"]] = "contour",
        xlim: Op[Tuple[float, float]] = None,
        central_longitude: Op[float] = None,
    ) -> Tuple[Tuple[Fig, ...], Tuple[Ax, ...]]:
        if variable not in ("cor", "reg"):
            raise ValueError(f"Expected `cor` or `reg` for argument `variable`, but got {variable}")
        if plot_type not in ("contour", "pcolor"):
            raise ValueError(f"Expected `contour` or `pcolor` for argument `plot_type`, but got {plot_type}")
        nm = nm if nm is not None else self.nm
        if nm > self.nm:
            raise ValueError(f"Can not draw more modes ({nm}) than the ones used for then methodology ({self.nm})")
        if signs is not None:
            if len(signs) != nm:
                raise TypeError(f'Expected signs to be a sequence of the same length as number of modes ({nm})')

        figsize = sp_pr._calculate_figsize(1, maxwidth=sp_pr.MAX_WIDTH, maxheight=sp_pr.MAX_HEIGHT) if figsize is None else figsize

        # created_figs = []
        #
        # fig1 = plt.figure(figsize=figsize)
        # ax = fig1.add_subplot()
        # ax.set_title("fvar")
        # ax.plot(self.fvar, color="black", linewidth=2)
        # ax.grid()
        #
        # created_figs.append(fig, (ax, ), "fvar.png")


        figs: List[Fig] = []
        axs: List[Tuple[Ax, ...]] = []
        nrows, ncols = 3, 3
        modes_per_page = nrows * ncols
        n_pages = math.ceil(nm / modes_per_page)
        for page_i in range(n_pages):
            mode0 = modes_per_page * page_i
            modef = min(nm - 1, modes_per_page * (page_i + 1) - 1)

            fig_i = plt.figure(figsize=figsize)
            region = self.ds.region
            # wratio = (region.lonf if region.lonf > region.lon0 else region.lonf + 360) - region.lon0
            gs = gridspec.GridSpec(nrows + 1, ncols, hspace=0.7, height_ratios=[1 for _ in range(nrows)] + [.1])

            central_longitude = central_longitude if central_longitude is not None else \
                sp_pr.get_central_longitude_from_region(region.lon0, region.lonf)
            axs_i = tuple(
                fig_i.add_subplot(gs[i, j], projection=ccrs.PlateCarree(central_longitude))
                for i in range(nrows)
                for j in range(ncols)
                if i*ncols + j <= modef - mode0
            )

            if variable == "cor":
                u, u_sig = self.cor, self.cor_sig
            elif variable == "reg":
                u, u_sig = self.reg, self.reg_sig
            else:
                assert False, "Unreachable"

            for i, ax in enumerate(axs_i):
                mode = mode0 + i
            
                lats = self.ds.lat
                lons = self.ds.lon

                ylim = sorted((lats.values[-1], lats.values[0]))

                if levels is None and plot_type != "scatter":
                    levels = np.linspace(-1, +1, 20)

                title = f'{self.ds.var} mode {mode + 1}. ' \
                        f'fvar={self.fvar[mode]*100:.01f}%'

                t = u[:, mode].transpose().reshape((len(lats), len(lons)))
                th = u_sig[:, mode].transpose().reshape((len(lats), len(lons)))

                if signs is not None:
                    if signs[mode]:
                        t *= -1

                add_cyclic_point = region.lon0 >= region.lonf
                im = sp_pr.plot_map(
                    t, lats, lons, fig_i, ax, title,
                    levels=levels, xlim=xlim, ylim=ylim, cmap=cmap, ticks=ticks,
                    colorbar=False, 
                    add_cyclic_point=add_cyclic_point and plot_type in ("contour", "pcolor"), 
                    plot_type=plot_type,
                )
                if mode == modef:
                    cb = fig_i.colorbar(im, cax=fig_i.add_subplot(gs[nrows, :]), orientation='horizontal', ticks=ticks,)
                    if ticks is None:
                        tick_locator = ticker.MaxNLocator(nbins=5, prune='both', symmetric=True)
                        cb.ax.xaxis.set_major_locator(tick_locator)
                    #cb.ax.xaxis.set_tick_params(rotation=20)
                hlons = lons
                th, hlons = add_cyclic_point_to_data(th, coord=hlons.values)
                ax.contourf(
                    hlons, lats, th, colors='none', hatches='..', extend='both',
                    transform=ccrs.PlateCarree()
                )

            fig_i.suptitle(
                f'{self.ds.var}: {region2str(self.ds.region)}, '
                f'Alpha: {self.alpha}',
                fontweight='bold'
            )

            fig_i.subplots_adjust(hspace=.4)

            
            if folder is None:
                folder = '.'
            if name is None:
                path = os.path.join(folder, f'weather_regimes-eof.png')
            else:
                path = os.path.join(folder, name)

            if n_pages > 1:
                path_name, path_extension = os.path.splitext(path)
                path = f"{path_name}_{page_i}{path_extension}"

            sp_pr._apply_flags_to_fig(
                fig_i, path,
                save_fig=save_fig,
                show_plot=show_plot,
                halt_program=False,
            )

            figs.append(fig_i)
            axs.append(axs_i)

        if show_plot and halt_program:
            plt.show(block=True)

        return tuple(figs), tuple(axs)

    @property
    def ds(self) -> Preprocess:
        return self._ds

    @classmethod
    def load(cls, prefix: str, folder: str = '.', 
             zip_file: Op[str] = None,
             *,
             ds: Op[Preprocess] = None,
             **attrs: Any) -> 'WeatherRegimes':
        if len(attrs) != 0:
            raise TypeError('Load only takes one keyword argument: ds')
        if ds is None:
            raise TypeError('To load an MCA object you must provide `ds` keyword argument')
        if type(ds) not in (Preprocess, ):
            raise TypeError(f'Unexpected type ({type(ds)}) for `ds`. Expected type `Preprocess`')

        self: WeatherRegimes = super().load(prefix, folder, zip_file)

        self._ds = ds
        return self

