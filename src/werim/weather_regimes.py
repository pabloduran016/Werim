from sys import modules
from spy4cast import add_cyclic_point_to_data

from werim.principal_components import PrincipalComponents
from ._common import *
from .preprocess import Preprocess
from ._log import *
from ._functions import *
from .principal_components import PrincipalComponents, plot_clusters_or_EOFs

class WeatherRegimes(sp_pr._Procedure):
    _pc: PrincipalComponents

    centroids: FArray  # k x nm
    index: FArray  # nt

    k: int

    VAR_NAMES = (
        'centroids', 'index',
        'k',
    )

    @property
    def var_names(self) -> Tuple[str, ...]:
        return self.VAR_NAMES

    def __init__(
        self,
        pc: PrincipalComponents,
        k: int,   # number of clusters
    ) -> None:
        log_info("Calculating Weather Regimes")
        here = time_from_here()

        self._pc = pc

        self.k = k
        self.centroids, self.index = scipy_cluster.vq.kmeans2(self.pc.PCs.T, self.k)

        log_debug(f' took {time_to_here(here):.03f} seconds', prefix="", info=False)

    @property
    def clusters(self) -> FArray:
        """Centoroid map for each cluster"""
        return np.dot(self.pc.cor, self.centroids.T)  # ns x k

    @property
    def clusters_sig(self) -> FArray:
        """Centroid significant map for each cluster"""
        return np.dot(self.pc.cor_sig, self.centroids.T)  # ns x k


    @property
    def composed_maps(self) -> FArray:
        """Composed maps for each cluster"""
        comp = np.zeros((self.ds.shape[0], self.k), np.float32)
        for i in range(comp.shape[0]):
            comp[:, i] = np.mean(self.ds.data[:, self.index == i], axis=1)
        return comp

    @property
    def ds(self) -> Preprocess:
        return self._pc.ds

    @property
    def pc(self) -> PrincipalComponents:
        return self._pc

    def plot_PCs(
        self,
        *,
        save_fig: bool = False,
        show_plot: bool = False,
        halt_program: bool = False,
        folder: Op[str] = None,
        name: Op[str] = None,
        figsize: Op[Tuple[float, float]] = None,
    ) -> Tuple[Tuple[Fig, ...], Tuple[Tuple[Ax, ...], ...]]:
        figsize = sp_pr._calculate_figsize(1, maxwidth=sp_pr.MAX_WIDTH, maxheight=sp_pr.MAX_HEIGHT) if figsize is None else figsize
        fig = plt.figure(figsize=figsize)
        fig.suptitle('PCs clusters', fontweight="bold")

        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel('PC1s')
        ax.set_ylabel('PC2s')
        ax.set_zlabel('PC3s')

        ax.grid()
        for nk in range(self.k):  
            ax.scatter3D(
                self.pc.PCs[0, self.index == nk],
                self.pc.PCs[1, self.index == nk],
                self.pc.PCs[2, self.index == nk],
                alpha=0.8,
            )

        if folder is None:
            folder = '.'
        if name is None:
            path = os.path.join(folder, f'wr-PCs_clusters.png')
        else:
            path = os.path.join(folder, name)

        sp_pr._apply_flags_to_fig(
            fig, path,
            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=False,
        )

        if show_plot and halt_program:
            plt.show(block=True)

        return (fig, ), ((ax, ), )


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
        plot_composed: bool = False,
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
        cluster_or_EOF = "cluster" if not plot_composed else "cluster-comp"
        return plot_clusters_or_EOFs(
            obj=self,
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
             pc: Op[PrincipalComponents] = None,
             **attrs: Any) -> 'WeatherRegimes':
        if len(attrs) != 0:
            raise TypeError('Load only takes one keyword argument: ds')
        if pc is None:
            raise TypeError('To load an WeatherRegimes object you must provide `pc` keyword argument')
        if type(pc) not in (PrincipalComponents, ):
            raise TypeError(f'Unexpected type ({type(pc)}) for `pc`. Expected type `PrincipalComponents`')

        self: WeatherRegimes = super().load(prefix, folder, zip_file)

        self._pc = pc
        return self

