from traits.api \
import HasTraits, Str, Int, Float, List, Bool, Property, Array, \
    Instance, File, Event, on_trait_change, cached_property, Tuple

from traitsui.api \
    import View, Item, TableEditor, VGroup, HSplit, Group, ModelView

from traitsui.menu import OKButton, MenuBar, Menu, Action
from mpl_figure_editor import MPLFigureEditor
from matplotlib.figure import Figure
import numpy as np
import pylab as p
from scipy.integrate import trapz

class DF(HasTraits):

    datafile = File()

    data = Property(Array, depends_on='datafile', data_changed=True)
    @cached_property
    def _get_data(self):
        return np.loadtxt(self.datafile, skiprows=1)

    x_range_enabled = Bool(False)

    x_min = Float(input_changed=True, enter_set=True, auto_set=False)
    @on_trait_change('data')
    def _x_min_update(self):
        self.x_min = self.data[:, 0].min()

    x_max = Float(input_changed=True, enter_set=True, auto_set=False)
    @on_trait_change('data')
    def _x_max_update(self):
        self.x_max = self.data[:, 0].max()

    x_mask = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_x_mask(self):
        x = self.data[:, 0]
        return np.logical_and(self.x_min <= x, x <= self.x_max)

    x = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_x(self):
        if self.x_range_enabled:
            return  self.data[:, 0][self.x_mask]
        else:
            return  self.data[:, 0]

    y = Property(Array, depends_on='+input_changed')
    @cached_property
    def _get_y(self):
        if self.x_range_enabled:
            return self.data[:, 1][self.x_mask]
        else:
            return self.data[:, 1]

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

    traits_view = View('datafile',
                       'N',
                       Item('x_range_enabled'),
                       Item('x_min', enabled_when='x_range_enabled'),
                       Item('x_max', enabled_when='x_range_enabled')
                       )


class DFView(HasTraits):
    df = Instance(DF)

    figure = Instance(Figure)
    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.add_axes([0.15, 0.15, 0.75, 0.75])  # [0.15, 0.15, 0.75, 0.75]
        return figure

    plot_xy = Bool(True, data_changed=True)
    plot_n_coeff = Bool(False, data_changed=True)
    plot_freq_coeff = Bool(False, data_changed=True)
    plot_freq_coeff_abs = Bool(False, data_changed=True)

    plot_title = Str(enter_set=True, auto_set=False, data_changed=True)
    label_fsize = Float(15, enter_set=True, auto_set=False, data_changed=True)
    tick_fsize = Float(15, enter_set=True, auto_set=False, data_changed=True)
    title_fsize = Float(15, enter_set=True, auto_set=False, data_changed=True)

    label_default = Bool(True, data_changed=True)
    x_label = Str('x', data_changed=True)
    x_limit_on = Bool(False, data_changed=True)
    x_limit = Tuple((0., 1.), data_changed=True)
    y_label = Str('y', data_changed=True)
    y_limit_on = Bool(False, data_changed=True)
    y_limit = Tuple((0., 1.), data_changed=True)

    data_changed = Event(True)

    @on_trait_change('df.+input_changed, df.+data_changed, +data_changed')
    def _redraw(self):
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
            axes.plot(df.x, df.y, color='blue', label='data')
            axes.plot(df.x, df.y_fourier, color='green', label='fourier')
            axes.legend()
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
            axes.legend()
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
            axes.legend()
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
            axes.legend()
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


#        if  str(self.model.get_data()) == 'False':
#            axes.set_title('File error' , \
#                            size = 'x-large', \
#                            weight = 'bold', position = (.5, 1.03))
#        else:
#            data = self.model.get_data()
#            #print self.model.get_data()
#            axes.hist(data, self.model.bins, normed = self.model.normed , \
#                        cumulative = self.model.cumulative , histtype = self.model.histtype, \
#                        align = self.model.align, orientation = self.model.orientation, \
#                        log = self.model.log, facecolor = 'blue', alpha = .3)
#            axes.set_xlabel('g(X) = R - E', weight = 'semibold')
#            axes.set_ylabel('Probability' , weight = 'semibold')
#            axes.set_title('Histogram ' + r'$g(X)$' , \
#                            size = 'x-large', \
#                            weight = 'bold', position = (.5, 1.03))
#            #axes.axis( [0, 4000, 0, 0.0012] )
#            axes.grid(True)
#            axes.set_axis_bgcolor(color = 'white')
#            axes.grid(color = 'gray', linestyle = '--', linewidth = 0.2, alpha = 0.75)
        self.data_changed = True

    traits_view = View(HSplit(
                              VGroup(
                                      Group(
                                            Item('df@', show_label=False),
                                            label='fourier settings',
                                            show_border=True,
                                            id='fourier.settings'
                                            ),
                                      Group(
                                            Item('plot_xy'),
                                            Item('plot_n_coeff'),
                                            Item('plot_freq_coeff'),
                                            Item('plot_freq_coeff_abs'),
                                            label='plot options',
                                            show_border=True,
                                            id='fourier.plot_options'
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
                                            id='fourier.plot_settings'
                                            ),
                                        id='fourier.control',
                                        dock='tab',
                                        label='settings',
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
                        id='fourier.view',
                        dock='tab',
                        resizable=True,
                        width=0.7,
                        height=0.7,
                        buttons=[OKButton]
                       )



def dfourier(x, y, T1=0, n=3):
    T1 = x.max() - x.min()
    Omega1 = np.pi * 2. / T1

    def get_a0(x, y, T1):
        return 1. / T1 * trapz(y, x)

    def cos_coeff(x, y, N):
        return 2. / T1 * trapz(y * np.cos(N * x * Omega1), x)

    def sin_coeff(x, y, N):
        return 2. / T1 * trapz(y * np.sin(N * x * Omega1), x)

    N = np.arange(n)[:, None] + 1
    a0 = get_a0(x, y, T1)
    cc = cos_coeff(x, y, N)
    ss = sin_coeff(x, y, N)
    print a0
    print cc
    print ss

    yy = a0 + np.sum(cc[:, None] * np.cos(N * x * Omega1) + ss[:, None] *
                     np.sin(N * x * Omega1), axis=0)

    freq = Omega1 * N / 2. / np.pi

    p.plot(x, y, label='function')
    p.plot(x, yy.T, label='fourier series')
    p.xlabel('time')
    p.ylabel('y')
    p.legend()

    p.figure()
    p.grid()
    p.vlines(N - 0.05, [0], cc, color='blue', label='cos')
    p.vlines(N + 0.05, [0], ss, color='green', label='sin')
    p.xlabel('N')
    p.ylabel('coeff')
    p.legend()

    p.figure()
    p.grid()
    p.vlines(freq, [0], cc, color='blue', label='cos')
    p.vlines(freq, [0], ss, color='green', label='sin')
    p.xlabel('frequency')
    p.ylabel('coeff')
    p.legend()

    p.show()

if __name__ == '__main__':
    df = DFView(df=DF(datafile='test.txt'))
    df.configure_traits()



#    data = np.loadtxt('jirka.txt', skiprows=1)
#    #data = np.loadtxt('2lidi-vyp1-acc.txt', skiprows=1)
#    x = data[:,0]
#    y = data[:,1]
#
#    sam = 1000
#    #x = np.linspace(-np.pi,np.pi,sam)
#    #y = np.ones(sam) * np.abs(x)- np.pi/2.
#    x = np.linspace(-np.pi,np.pi,sam)
#    y = np.ones(sam) * 10 * (x>=0)-5
#    y[0]=0
#    y[-1]=0
#    #x = np.linspace(-np.pi,np.pi,sam)
#    #y = np.ones(sam) * x
#    #x = np.linspace(0,6,sam)
#    #y = np.cos(2*np.pi*x)
#
#    dfourier(x,y, n=9)





