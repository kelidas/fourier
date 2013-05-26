from traits.api \
import HasTraits, Str, Int, Float, Bool, Property, Array, Trait, \
    Instance, File, Event, on_trait_change, cached_property, Tuple, Button, DelegatesTo

from traitsui.api \
    import View, Item, VGroup, HSplit, Group, UItem, HGroup, spring, Tabbed, \
    Label, FileEditor

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
        return  self.data.x[self.x_mask]

    y = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_y(self):
        return self.data.y[self.x_mask]

    N = Int(5, enter_set=True, auto_set=False, input_changed=True)

    N_arr = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_N_arr(self):
        return np.arange(self.N)[:, None] + 1

    T1 = Property(Float, depends_on='+input_changed')
    @cached_property
    def _get_T1(self):
        return float(self.x.max() - self.x.min())

    Omega1 = Property(Float, depends_on='+input_changed')
    @cached_property
    def _get_Omega1(self):
        return np.pi * 2.0 / self.T1

    a0 = Property(Float, depends_on='+input_changed')
    @cached_property
    def _get_a0(self):
        a0 = 1. / self.T1 * trapz(self.y, self.x)
        print 'a_0 =', a0
        return a0

    cos_coeff = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_cos_coeff(self):
        # return 2. / self.T1 * trapz(self.y * np.cos(self.N_arr * self.x * self.Omega1), self.x)
        res = np.zeros(self.N)
        for i, n in enumerate(self.N_arr):
            res[i] = 2. / self.T1 * trapz(self.y * np.cos(n * self.x * self.Omega1), self.x)
        print 'cos coefficients =', res
        return res

    sin_coeff = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_sin_coeff(self):
        # return 2. / self.T1 * trapz(self.y * np.sin(self.N_arr * self.x * self.Omega1), self.x)
        res = np.zeros(self.N)
        for i, n in enumerate(self.N_arr):
            res[i] = 2. / self.T1 * trapz(self.y * np.sin(n * self.x * self.Omega1), self.x)
        print 'sin coefficients =', res
        return res

    energy = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_energy(self):
        np.savetxt('energ.txt', np.sqrt(self.sin_coeff ** 2 + self.cos_coeff ** 2))
        return np.sqrt(self.sin_coeff ** 2 + self.cos_coeff ** 2)

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
        np.savetxt('f_fourier.txt', self.Omega1 * self.N_arr / 2. / np.pi)
        return self.Omega1 * self.N_arr / 2. / np.pi


class ControlPanel(HasTraits):
    datafile = File(auto_set=False, enter_set=True)

    data = Instance(Data, ())

    load_data = Button()
    def _load_data_fired(self):
        self.data.data = np.loadtxt(self.datafile, skiprows=1)
        self.df.data = self.data

    df = Instance(DF, ())

    x_range_enabled = Bool(False)
    def _x_range_enabled_changed(self):
        if self.x_range_enabled == False:
            self.df._x_min_update()

    x_min = DelegatesTo('df')
    x_max = DelegatesTo('df')
    N = DelegatesTo('df')

    view = View(
                VGroup(
                       HGroup(
                       Item('datafile', springy=True,
                             id='control_panel.datafile_hist'),
                       UItem('load_data', id='control_panel.load_data'),
                       id='control_panel.datafile_2'
                       ),
                     Group(
                           Item('N', id='control_panel.N'),
                           '_',
                           Label('Set range of one period'),
                           HGroup(
                               UItem('x_range_enabled', id='control_panel.x_range_enabled'),
                               Item('x_min', enabled_when='x_range_enabled', id='control_panel.x_min'),
                               Item('x_max', enabled_when='x_range_enabled', id='control_panel.x_max'),
                               ),
                           label='parameters of transformation',
                           show_border=True,
                           id='control_panel.parameters',
                               ),
                       ),
                id='control_panel.view',
                )


class MainWindow(HasTraits):
    panel = Instance(ControlPanel, ())

    df = DelegatesTo('panel')

    plot_fourier_series = Bool(True)
    plot_data = Bool(True)

    plot_type = Trait('0_plot_xy', {'0_plot_xy':0,
                                  '1_plot_n_coeff':1,
                                  '2_plot_freq_coeff':2,
                                  '3_plot_freq_coeff_abs':3,
                                  '4_plot_freq_energy':4})

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

    figure = Instance(Figure)
    def _figure_default(self):
        figure = Figure(tight_layout=True)
        figure.add_subplot(111)
        # figure.add_axes([0.15, 0.15, 0.75, 0.75])
        return figure

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
        if self.plot_type_ == 0:
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

        if self.plot_type_ == 1:
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

        if self.plot_type_ == 2:
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

        if self.plot_type_ == 3:
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

        if self.plot_type_ == 4:
            axes.plot(df.freq, df.energy, 'k-', label='energ')
            axes.legend(loc='best')
            axes.set_title(self.plot_title, fontsize=title_fsize)
            if self.label_default:
                self.x_label = 'freq'
                self.y_label = 'energy'
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
                                        Item('panel@', show_label=False, id='fourier.panel'),
                                          Group(
                                                Item('plot_data'),
                                                Item('plot_fourier_series'),
                                                '_',
                                                Item('plot_type'),
                                                label='plot options',
                                                show_border=True,
                                                id='fourier.plot_options'
                                                ),
                                        UItem('draw', label='calculate and draw'),
                                         label='fourier',
                                         id='fourier.fourier'
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
                                    Item('figure', editor=MPLFigureEditor(),
                                    resizable=True, show_label=False),
                                    label='Plot sheet',
                                    id='fourier.figure',
                                    dock='tab',
                                    ),
                                 ),
                        title='Fourier series',
                        id='main_window.view',
                        resizable=True,
                        width=0.7,
                        height=0.7,
                        buttons=[OKButton]
                       )


if __name__ == '__main__':
    df = MainWindow()
    df.configure_traits()








