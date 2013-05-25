from traits.api \
import HasTraits, Str, Int, Float, Bool, Property, Array, \
    Instance, File, Event, on_trait_change, cached_property, Tuple, Button, DelegatesTo

from traitsui.api \
    import View, Item, VGroup, HSplit, Group, UItem, HGroup, spring, Tabbed, Label

from traitsui.menu import OKButton
from mpl_figure_editor import MPLFigureEditor
from matplotlib.figure import Figure
import numpy as np
import pylab as p
from scipy.integrate import trapz


class Data(HasTraits):
    data = Array()

    x = Array()
    @on_trait_change('data')
    def _x_set(self):
        self.x = self.data[:, 0]

    y = Array()
    @on_trait_change('data')
    def _y_set(self):
        self.y = self.data[:, 1]


class DF(HasTraits):
    data = Instance(Data)

    x_range_enabled = Bool(False)

    x_min = Float(enter_set=True, auto_set=False, input_changed=True)
    @on_trait_change('data.data')
    def _x_min_update(self):
        self.x_min = self.data.x.min()

    x_max = Float(enter_set=True, auto_set=False, input_changed=True)
    @on_trait_change('data.data')
    def _x_max_update(self):
        self.x_max = self.data.x.max()

    x_mask = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_x_mask(self):
        x = self.data.x
        return np.logical_and(self.x_min <= x, x <= self.x_max)

    x = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_x(self):
        if self.x_range_enabled:
            return  self.data.x[self.x_mask]
        else:
            return  self.data.x

    y = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_y(self):
        if self.x_range_enabled:
            return self.data.y[self.x_mask]
        else:
            return self.data.y

    N = Int(5, enter_set=True, auto_set=False, input_changed=True)

    N_arr = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_N_arr(self):
        return np.arange(self.N)[:, None] + 1

    T1 = Property(Float, depends_on='+input_changed')
    @cached_property
    def _get_T1(self):
        return self.x.max() - self.x.min()

    Omega1 = Property(Float, depends_on='+input_changed')
    @cached_property
    def _get_Omega1(self):
        return np.pi * 2. / self.T1

    a0 = Property(Float, depends_on='+input_changed')
    @cached_property
    def _get_a0(self):
        return 1. / self.T1 * trapz(self.y, self.x)

    cos_coeff = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_cos_coeff(self):
        # return 2. / self.T1 * trapz(self.y * np.cos(self.N_arr * self.x * self.Omega1), self.x)
        res = np.zeros(self.N)
        for i, n in enumerate(self.N_arr):
            res[i] = 2. / self.T1 * trapz(self.y * np.cos(n * self.x * self.Omega1), self.x)
        return res

    sin_coeff = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_sin_coeff(self):
        # return 2. / self.T1 * trapz(self.y * np.sin(self.N_arr * self.x * self.Omega1), self.x)
        res = np.zeros(self.N)
        for i, n in enumerate(self.N_arr):
            res[i] = 2. / self.T1 * trapz(self.y * np.sin(n * self.x * self.Omega1), self.x)
        return res

    y_fourier = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_y_fourier(self):
#        return self.a0 + np.sum(self.cos_coeff[:, None] * np.cos(self.N_arr * self.x *
#                           self.Omega1) + self.sin_coeff[:, None] *
#                          np.sin(self.N_arr * self.x * self.Omega1), axis=0)
        res = np.zeros_like(self.x)
        for i, x in enumerate(self.x):
            res[i] = self.a0 + np.sum(self.cos_coeff[:, None] * np.cos(self.N_arr * x *
                           self.Omega1) + self.sin_coeff[:, None] *
                          np.sin(self.N_arr * x * self.Omega1))
        return res

    freq = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_freq(self):
        return self.Omega1 * self.N_arr / 2. / np.pi

    traits_view = View(
                       Group(
                           Item('N'),
                           '_',
                           Label('Set range of one period'),
                           HGroup(
                               UItem('x_range_enabled'),
                               Item('x_min', enabled_when='x_range_enabled'),
                               Item('x_max', enabled_when='x_range_enabled'),
                               ),
                           label='configure parameters for analysis',
                           show_border=True,
                           id='df.view',
                           ),
                       )


class ControlPanel(HasTraits):
    datafile = File(auto_set=False, enter_set=True)

    data = Instance(Data, ())

    load_data = Button()
    def _load_data_fired(self):
        self.data.data = np.loadtxt(self.datafile, skiprows=1)
        self.df.data = self.data

    df = Instance(DF, ())

    view = View(
                HGroup(
                       Item('datafile', springy=True, id='control_panel.datafile'),
                       UItem('load_data'),
                       ),
                UItem('df@'),
                id='control_panel.view',
                )


class MainWindow(HasTraits):
    panel = Instance(ControlPanel, ())

    df = DelegatesTo('panel')

    figure = Instance(Figure)
    def _figure_default(self):
        figure = Figure(tight_layout=True)
        figure.add_subplot(111)
        # figure.add_axes([0.15, 0.15, 0.75, 0.75])
        return figure

    plot_fourier_series = Bool(True)
    plot_data = Bool(True)
    plot_xy = Bool(True)
    plot_n_coeff = Bool(False)
    plot_freq_coeff = Bool(False)
    plot_freq_coeff_abs = Bool(False)

    plot_title = Str(enter_set=True, auto_set=False, changed=True)
    label_fsize = Float(15, enter_set=True, auto_set=False, changed=True)
    tick_fsize = Float(15, enter_set=True, auto_set=False, changed=True)
    title_fsize = Float(15, enter_set=True, auto_set=False, changed=True)

    label_default = Bool(True)
    x_label = Str('x', changed=True)
    x_limit_on = Bool(False)
    x_limit = Tuple((0., 1.), changed=True)
    y_label = Str('y', changed=True)
    y_limit_on = Bool(False)
    y_limit = Tuple((0., 1.), changed=True)

    @on_trait_change('+changed')
    def _redraw(self):
        self._draw_fired()

    draw = Button
    def _draw_fired(self):
        figure = self.figure
        axes = figure.axes[0]
        axes.clear()
        # self.x_limit = (axes.axis()[0], axes.axis()[1])
        # self.y_limit = (axes.axis()[2], axes.axis()[3])
        df = self.df

        label_fsize = self.label_fsize
        tick_fsize = self.tick_fsize
        title_fsize = self.title_fsize
        if self.plot_xy:
            if self.plot_data and self.plot_fourier_series == False:
                axes.plot(df.x, df.y, color='blue', label='data')
            elif self.plot_fourier_series and self.plot_data == False:
                axes.plot(df.x, df.y_fourier, color='green', label='fourier')
            else:
                axes.plot(df.x, df.y, color='blue', label='data')
                axes.plot(df.x, df.y_fourier, color='green', label='fourier')
            axes.legend(loc='best')
            axes.grid()
            if self.label_default:
                self.x_label = 'x'
                self.y_label = 'y'
            axes.set_xlabel(self.x_label, fontsize=label_fsize)
            axes.set_ylabel(self.y_label, fontsize=label_fsize)
            axes.set_title(self.plot_title, fontsize=title_fsize)
            if self.x_limit_on:
                axes.set_xlim(self.x_limit)
            if self.y_limit_on:
                axes.set_ylim(self.y_limit)
            p.setp(axes.get_xticklabels(), fontsize=tick_fsize, position=(0, -.01))  # position - posun od osy x
            p.setp(axes.get_yticklabels(), fontsize=tick_fsize)

        if self.plot_n_coeff:
            axes.vlines(df.N_arr - 0.05, [0], df.cos_coeff, color='blue', label='cos')
            axes.vlines(df.N_arr + 0.05, [0], df.sin_coeff, color='green', label='sin')
            axes.legend(loc='best')
            axes.grid()
            if self.label_default:
                self.x_label = 'n'
                self.y_label = 'coeff'
            axes.set_title(self.plot_title, fontsize=title_fsize)
            axes.set_xlabel(self.x_label, fontsize=label_fsize)
            axes.set_ylabel(self.y_label, fontsize=label_fsize)
            if self.x_limit_on:
                axes.set_xlim(self.x_limit)
            if self.y_limit_on:
                axes.set_ylim(self.y_limit)
            p.setp(axes.get_xticklabels(), fontsize=tick_fsize, position=(0, -.01))  # position - posun od osy x
            p.setp(axes.get_yticklabels(), fontsize=tick_fsize)

        if self.plot_freq_coeff:
            axes.vlines(df.freq, [0], df.cos_coeff, color='blue', label='cos')
            axes.vlines(df.freq, [0], df.sin_coeff, color='green', label='sin')
            axes.legend(loc='best')
            axes.grid()
            if self.label_default:
                self.x_label = 'freq'
                self.y_label = 'coeff'
            axes.set_title(self.plot_title, fontsize=title_fsize)
            axes.set_xlabel(self.x_label, fontsize=label_fsize)
            axes.set_ylabel(self.y_label, fontsize=label_fsize)
            if self.x_limit_on:
                axes.set_xlim(self.x_limit)
            if self.y_limit_on:
                axes.set_ylim(self.y_limit)
            p.setp(axes.get_xticklabels(), fontsize=tick_fsize, position=(0, -.01))  # position - posun od osy x
            p.setp(axes.get_yticklabels(), fontsize=tick_fsize)

        if self.plot_freq_coeff_abs:
            axes.vlines(df.freq, [0], np.abs(df.cos_coeff), color='blue', label='cos')
            axes.vlines(df.freq, [0], np.abs(df.sin_coeff), color='green', label='sin')
            axes.legend(loc='best')
            axes.set_title(self.plot_title, fontsize=title_fsize)
            if self.label_default:
                self.x_label = 'freq'
                self.y_label = 'coeff'
            axes.set_xlabel(self.x_label, fontsize=label_fsize)
            axes.set_ylabel(self.y_label, fontsize=label_fsize)
            y_val = np.abs(np.hstack((df.cos_coeff, df.sin_coeff))).max()
            axes.set_ybound((0, y_val * 1.05))
            if self.x_limit_on:
                axes.set_xlim(self.x_limit)
            if self.y_limit_on:
                axes.set_ylim(self.y_limit)
            p.setp(axes.get_xticklabels(), fontsize=tick_fsize, position=(0, -.01))  # position - posun od osy x
            p.setp(axes.get_yticklabels(), fontsize=tick_fsize)

        self.figure.canvas.draw()

    traits_view = View(HSplit(
                              Tabbed(
                                  Group(
                                        Item('panel@', show_label=False),
                                          Group(
                                                Item('plot_data'),
                                                Item('plot_fourier_series'),
                                                '_',
                                                Item('plot_xy'),
                                                Item('plot_n_coeff'),
                                                Item('plot_freq_coeff'),
                                                Item('plot_freq_coeff_abs'),
                                                label='plot options',
                                                show_border=True,
                                                id='fourier.plot_options'
                                                ),
                                         label='fourier',
                                         dock='tab',
                                         ),
                                 Group(
                                        Item('plot_title', label='title'),
                                        Item('title_fsize', label='title fontsize'),
                                        Item('label_fsize', label='label fontsize'),
                                        Item('tick_fsize', label='tick fontsize'),
                                        Item('_'),
                                        Item('x_limit_on'),
                                        Item('x_limit', label='x limit - plot', enabled_when='x_limit_on'),
                                        Item('_'),
                                        Item('y_limit_on'),
                                        Item('y_limit', label='y limit - plot', enabled_when='y_limit_on'),
                                        Item('_'),
                                        Item('label_default'),
                                        Item('x_label', enabled_when='label_default==False'),
                                        Item('y_label', enabled_when='label_default==False'),
                                        label='plot settings',
                                        show_border=True,
                                        id='fourier.plot_settings',
                                    dock='tab',
                                  ),
                                ),
                              VGroup(
                                     UItem('draw'),
                                    Item('figure', editor=MPLFigureEditor(),
                                    resizable=True, show_label=False),
                                    label='Plot sheet',
                                    id='fourier.figure',
                                    dock='tab',
                                    ),
                                 ),
                        title='Fourier series',
                        id='main_window.view',
                        dock='tab',
                        resizable=True,
                        width=0.7,
                        height=0.7,
                        buttons=[OKButton]
                       )


if __name__ == '__main__':
    df = MainWindow()
    df.configure_traits()







