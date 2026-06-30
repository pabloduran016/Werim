from sys import modules
from spy4cast import add_cyclic_point_to_data
from ._common import *
from .preprocess import Preprocess
from ._log import *
from ._functions import *

class WeatherRegimes(sp_pr._Procedure):
    eof: FArray
    singular_values: FArray
    eof2: FArray
    fvar: FArray

    cor: FArray
    cor_sig: FArray
    reg: FArray
    reg_sig: FArray
    pvalue: FArray
    PCs: FArray

    centroids: FArray  # k x nm
    index: FArray  # ns

    alpha: float
    nm: int
    k: int

    VAR_NAMES = (
        'eof', 'eof2', 'singular_values',
        'fvar', 
        'cor', 'cor_sig',
        'reg', 'reg_sig',
        'pvalue', 'PCs',
        'centroids', 'index',
        'alpha', 'nm', 'k',
    )

    @property
    def var_names(self) -> Tuple[str, ...]:
        return self.VAR_NAMES

    def __init__(
        self,
        ds: Preprocess,
        nm: int,
        k: int,   # number of clusters
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

            
        self.k = k
        self.centroids, self.index = scipy_cluster.vq.kmeans2((self.PCs.T ), self.k)

        log_debug(f'    Took: {time_to_here(here):.03f} seconds', prefix="", info=False)

    @property
    def clusters(self) -> FArray:
        """Centoroid map for each cluster"""
        return np.dot(self.cor, self.centroids.T)  # k x ns

    @property
    def clusters_sig(self) -> FArray:
        """Centroid significant map for each cluster"""
        return np.dot(self.cor_sig, self.centroids.T)  # k x ns

    @property
    def ds(self) -> Preprocess:
        return self._ds

    def plot_PCs(
        self,
        *,
        save_fig: bool = False,
        show_plot: bool = False,
        halt_program: bool = False,
        folder: Op[str] = None,
        name: Op[str] = None,
    ) -> Tuple[Tuple[Fig, ...], Tuple[Tuple[Ax, ...], ...]]:

        figs: List[Fig] = []
        axs: List[Tuple[Ax, ...]] = []

        figsize = sp_pr._calculate_figsize(1 / 2, maxwidth=sp_pr.MAX_WIDTH, maxheight=sp_pr.MAX_HEIGHT)
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Principal Components", fontweight="bold")

        gs = gridspec.GridSpec(2, 4, hspace=0.7)
        axes = (
            fig.add_subplot(gs[0, :2]),
            fig.add_subplot(gs[0, 2:]),
            fig.add_subplot(gs[1, 1:3]),
        )

        colors = ["red", "green", "blue"]
        for i, ax in enumerate(axes):
            ax.plot(self.ds.time, self.PCs[i, :], color=colors[i%len(colors)])
            ax.set_title(f"{self.ds.var} mode {i + 1}. fvar={self.fvar[i]*100:.01f}%")
            ax.set_xlabel("time")
            ax.grid()

        if folder is None:
            folder = '.'
        if name is None:
            path = os.path.join(folder, f'weather_regimes-PCs_series.png')
        else:
            path = os.path.join(folder, name)

        sp_pr._apply_flags_to_fig(
            fig, path,
            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=False,
        )
        figs.append(fig)
        axs.append(axes)

        figsize = sp_pr._calculate_figsize(1, maxwidth=sp_pr.MAX_WIDTH, maxheight=sp_pr.MAX_HEIGHT)
        fig = plt.figure(figsize=figsize)
        fig.suptitle('PCs clusters', fontweight="bold")

        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel('PC1s')
        ax.set_ylabel('PC2s')
        ax.set_zlabel('PC3s')

        ax.grid()
        for nk in range(self.k):  
            ax.scatter3D(
                self.PCs[0, self.index == nk],
                self.PCs[1, self.index == nk],
                self.PCs[2, self.index == nk],
                alpha=0.8,
            )

        if folder is None:
            folder = '.'
        if name is None:
            path = os.path.join(folder, f'weather_regimes-PCs_clusters.png')
        else:
            path = os.path.join(folder, name)

        sp_pr._apply_flags_to_fig(
            fig, path,
            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=False,
        )
        figs.append(fig)
        axs.append((ax, ))

        if show_plot and halt_program:
            plt.show(block=True)

        return tuple(figs), tuple(axs)

    def plot(
        self,
        *,
        cluster_or_EOF: Op[Literal["cluster", "EOF"]] = None,
        save_fig: bool = False,
        show_plot: bool = False,
        halt_program: bool = False,
        folder: Op[str] = None,
        name: Op[str] = None,
        figsize: Op[Tuple[float, float]] = None,
        cmap: str = 'bwr',
        n: Op[int] = None,
        n_per_page: Op[int] = None,
        nrows: Op[int] = None,
        ncols: Op[int] = None,
        cor_or_reg: Op[Literal["cor", "reg"]] = "cor",
        signs: Op[Sequence[bool]] = None,
        levels: Op[
            Union[npt.NDArray[np.float32], Sequence[float], bool]
        ] = None,
        ticks: Op[Union[npt.NDArray[np.float32], Sequence[float]]] = None,
        plot_type: Literal["contour", "pcolor"] = "contour",
        xlim: Op[Tuple[float, float]] = None,
        central_longitude: Op[float] = None,
    ) -> Tuple[Tuple[Fig, ...], Tuple[Tuple[Ax, ...], ...]]:
        if cluster_or_EOF is None:
            raise ValueError(f"Missing keyword argument `cluster_or_EOF`. Expected `cluster` or `EOF`.")
        return plot_clusters_or_EOFs(
            wr=self,
            cluster_or_EOF=cluster_or_EOF,
            n=n,

            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=halt_program,
            folder=folder,
            name=name,

            figsize=figsize,
            cmap=cmap,
            n_per_page=n_per_page,

            nrows=nrows,
            ncols=ncols,

            cor_or_reg=cor_or_reg,

            signs=signs,
            levels=levels,
            ticks=ticks,
            plot_type=plot_type,
            xlim=xlim,
            central_longitude=central_longitude,
        )

    def plot_clusters(
        self,
        *,
        save_fig: bool = False,
        show_plot: bool = False,
        halt_program: bool = False,
        folder: Op[str] = None,
        name: Op[str] = None,
        figsize: Op[Tuple[float, float]] = None,
        cmap: str = 'bwr',
        n_clusters: Op[int] = None,
        n_per_page: Op[int] = None,
        nrows: Op[int] = None,
        ncols: Op[int] = None,
        signs: Op[Sequence[bool]] = None,
        levels: Op[
            Union[npt.NDArray[np.float32], Sequence[float], bool]
        ] = None,
        ticks: Op[Union[npt.NDArray[np.float32], Sequence[float]]] = None,
        plot_type: Literal["contour", "pcolor"] = "contour",
        xlim: Op[Tuple[float, float]] = None,
        central_longitude: Op[float] = None,
    ) -> Tuple[Tuple[Fig, ...], Tuple[Tuple[Ax, ...], ...]]:
        n = self.k if n_clusters is None else n_clusters
        cor_or_reg = None
        cluster_or_EOF = "cluster"
        return plot_clusters_or_EOFs(
            wr=self,
            cluster_or_EOF=cluster_or_EOF,
            n=n,

            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=halt_program,
            folder=folder,
            name=name,

            figsize=figsize,
            cmap=cmap,
            n_per_page=n_per_page,

            nrows=nrows,
            ncols=ncols,

            cor_or_reg=cor_or_reg,

            signs=signs,
            levels=levels,
            ticks=ticks,
            plot_type=plot_type,
            xlim=xlim,
            central_longitude=central_longitude,
        )

    def plot_EOFs(
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
        modes_per_page: Op[int] = None,
        nrows: Op[int] = None,
        ncols: Op[int] = None,
        cor_or_reg: Literal["cor", "reg"] = "cor",
        signs: Op[Sequence[bool]] = None,
        levels: Op[
            Union[npt.NDArray[np.float32], Sequence[float], bool]
        ] = None,
        ticks: Op[Union[npt.NDArray[np.float32], Sequence[float]]] = None,
        plot_type: Literal["contour", "pcolor"] = "contour",
        xlim: Op[Tuple[float, float]] = None,
        central_longitude: Op[float] = None,
    ) -> Tuple[Tuple[Fig, ...], Tuple[Tuple[Ax, ...], ...]]:
        n = nm if nm is not None else self.nm
        if n > self.nm:
            raise ValueError(f"Can not draw more modes ({nm}) than the ones used for then methodology ({self.nm})")
        cluster_or_EOF = "EOF"
        n_per_page = modes_per_page
        return plot_clusters_or_EOFs(
            wr=self,
            cluster_or_EOF=cluster_or_EOF,
            n=n,

            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=halt_program,
            folder=folder,
            name=name,

            figsize=figsize,
            cmap=cmap,
            n_per_page=n_per_page,

            nrows=nrows,
            ncols=ncols,

            cor_or_reg=cor_or_reg,

            signs=signs,
            levels=levels,
            ticks=ticks,
            plot_type=plot_type,
            xlim=xlim,
            central_longitude=central_longitude,
        )

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


def plot_clusters_or_EOFs(
    *,
    wr,
    cluster_or_EOF: Literal["cluster", "EOF"],
    n: int,  # k or nm

    save_fig: bool,
    show_plot: bool,
    halt_program: bool,
    folder: Op[str],
    name: Op[str],

    figsize: Op[Tuple[float, float]],
    cmap: str,
    n_per_page: Op[int],

    nrows: Op[int],
    ncols: Op[int],

    cor_or_reg: Op[Literal["cor", "reg"]],

    signs: Op[Sequence[bool]],
    levels: Op[
        Union[npt.NDArray[np.float32], Sequence[float], bool]
    ],
    ticks: Op[Union[npt.NDArray[np.float32], Sequence[float]]],
    plot_type: Literal["contour", "pcolor"],
    xlim: Op[Tuple[float, float]],
    central_longitude: Op[float],
) -> Tuple[Tuple[Fig, ...], Tuple[Tuple[Ax, ...], ...]]:
    if cluster_or_EOF == "cluster":
        if cor_or_reg is not None:
            raise ValueError(f"Unexpected keyword argument `cor_or_reg`, to plot {cluster_or_EOF}")
    elif cluster_or_EOF == "EOF":
        if cor_or_reg not in ("cor", "reg"):
            raise ValueError(f"Expected `cor` or `reg` for argument `cor_or_reg`, but got {cor_or_reg}")
    else:
        raise ValueError(f"Expected `cluster` or `EOF` for argument `cluster_or_EOF`, but got {cluster_or_EOF}")

    if plot_type not in ("contour", "pcolor"):
        raise ValueError(f"Expected `contour` or `pcolor` for argument `plot_type`, but got {plot_type}")
    if signs is not None:
        if len(signs) != n:
            raise TypeError(f'Expected signs to be a sequence of the same length as number of clusters /modes ({n})')

    figs: List[Fig] = []
    axs: List[Tuple[Ax, ...]] = []

    ncols = min(4, math.ceil(np.sqrt(n))) if ncols is None else ncols
    nrows = min(3, math.ceil(n / ncols)) if nrows is None else nrows
    n_per_page = nrows * ncols if n_per_page is None else n_per_page
    region = wr.ds.region
    wratio = (region.lonf if region.lonf > region.lon0 else region.lonf + 360) - region.lon0
    ratio = nrows * (region.latf - region.lat0) / (ncols * wratio)
    figsize = sp_pr._calculate_figsize(ratio, maxwidth=sp_pr.MAX_WIDTH, maxheight=sp_pr.MAX_HEIGHT) if figsize is None else figsize
    gs = gridspec.GridSpec(
        nrows + 1, ncols, 
        height_ratios=[1]*nrows + [0.15],
        hspace=0.7)

    n_pages = math.ceil(n / n_per_page)
    for page_i in range(n_pages):
        mode0 = n_per_page * page_i
        modef = min(n - 1, n_per_page * (page_i + 1) - 1)

        fig_i = plt.figure(figsize=figsize)
        # wratio = (region.lonf if region.lonf > region.lon0 else region.lonf + 360) - region.lon0

        central_longitude = central_longitude if central_longitude is not None else \
            sp_pr.get_central_longitude_from_region(region.lon0, region.lonf)
        axs_i = tuple(
            fig_i.add_subplot(gs[i, j], projection=ccrs.PlateCarree(central_longitude))
            for i in range(nrows)
            for j in range(ncols)
            if i*ncols + j <= modef - mode0
        )

        if cluster_or_EOF == "cluster":
            u, u_sig = wr.clusters, wr.clusters_sig
        elif cluster_or_EOF == "EOF":
            if cor_or_reg == "cor":
                u, u_sig = wr.cor, wr.cor_sig
            elif cor_or_reg == "reg":
                u, u_sig = wr.reg, wr.reg_sig
            else:
                assert False, "Unreachable"
        else:
            assert False, "Unreachable"


        for i, ax in enumerate(axs_i):
            mode = mode0 + i
        
            lats = wr.ds.lat
            lons = wr.ds.lon

            ylim = sorted((lats.values[-1], lats.values[0]))

            if levels is None and plot_type != "scatter":
                levels = np.linspace(-1, +1, 20)

            if cluster_or_EOF == "cluster":
                title = f'{wr.ds.var} cluster {mode + 1}'
            elif cluster_or_EOF == "EOF":
                title = f'{wr.ds.var} mode {mode + 1}. fvar={wr.fvar[mode]*100:.01f}%'
            else:
                assert False, "Unreachable"

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
            f'{cluster_or_EOF}s ({wr.ds.var}): {region2str(wr.ds.region)}, '
            f'Alpha: {wr.alpha}',
            fontweight='bold'
        )
        fig_i.subplots_adjust(hspace=.4)
        
        if folder is None:
            folder = '.'
        if name is None:
            path = os.path.join(folder, f'weather_regimes-{cluster_or_EOF}s.png')
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

