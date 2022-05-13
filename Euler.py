import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import numpy as np

# 万有引力常数
G = 1
#三个物体的质量
ms0 = 6
me0 = 8
mm0 = 6
#x方向初始位置
xs0 = 0
xe0 = 3
xm0 = 5
#y方向初始位置
ys0 = 1
ye0 = -1
ym0 = 0
#初始速度
p1 = .35561
p2 = .46410
vxs0 = p1
vxe0 = -p2
vxm0 = -p1
vys0 = -2*p2
vye0 = p1
vym0 = 2

# 时间步长
tau = .001
T = np.arange(0, 25, tau)

speed = 10
border = 10


# 两点之间的距离
def r(xi, xj, yi, yj):
    return ((xi - xj) ** 2 + (yi - yj) ** 2) ** .5


# 点（x，y）受到其余两个物体合力的x分力所产生的加速度
def hx(mi, mj, x, xi, xj, y, yi, yj):
    return - G * mi * ((x - xi) / r(x, xi, y, yi) ** 3) - G * mj * ((x - xj) / r(x, xj, y, yj) ** 3)


# 点（x，y）受到其余两个物体合力的y分力所产生的加速度
def hy(mi, mj, y, yi, yj, x, xi, xj):
    return - G * mi * ((y - yi) / r(y, yi, x, xi) ** 3) - G * mj * ((y - yj) / r(y, yj, x, xj) ** 3)


# 龙格库塔函数
def Euler(ms, me, mm, xs0, xe0, xm0, vxs0, vxe0, vxm0, ys0, ye0, ym0, vys0, vye0, vym0):
    xs = [xs0]
    xe = [xe0]
    xm = [xm0]
    vxs = [vxs0]
    vxe = [vxe0]
    vxm = [vxm0]
    ys = [ys0]
    ye = [ye0]
    ym = [ym0]
    vys = [vys0]
    vye = [vye0]
    vym = [vym0]

    for _ in T:
        # 各个物体受到的x方向加速度
        asx = hx(me, mm, xs[-1], xe[-1], xm[-1], ys[-1], ye[-1], ym[-1])
        aex = hx(ms, mm, xe[-1], xs[-1], xm[-1], ye[-1], ys[-1], ym[-1])
        amx = hx(me, ms, xm[-1], xe[-1], xs[-1], ym[-1], ye[-1], ys[-1])

        # 计算得出x方向速度大小
        vsx1 = vxs[-1] + asx * tau
        vex1 = vxe[-1] + aex * tau
        vmx1 = vxm[-1] + amx * tau

        # 用步长乘以速度，得出x位移大小
        xs.append(xs[-1] + tau * vsx1)
        xe.append(xe[-1] + tau * vex1)
        xm.append(xm[-1] + tau * vmx1)
        vxs.append(vsx1)
        vxe.append(vex1)
        vxm.append(vmx1)

        asy = hy(me, mm, ys[-1], ye[-1], ym[-1], xs[-1], xe[-1], xm[-1])
        aey = hy(ms, mm, ye[-1], ys[-1], ym[-1], xe[-1], xs[-1], xm[-1])
        amy = hy(me, ms, ym[-1], ye[-1], ys[-1], xm[-1], xe[-1], xs[-1])

        vsy1 = vys[-1] + asy * tau
        vey1 = vye[-1] + aey * tau
        vmy1 = vym[-1] + amy * tau
        ys.append(ys[-1] + tau * vsy1)
        ye.append(ye[-1] + tau * vey1)
        ym.append(ym[-1] + tau * vmy1)
        vys.append(vsy1)
        vye.append(vey1)
        vym.append(vmy1)

    return xs, xe, xm, ys, ye, ym

# 画板设置
fig = plt.figure(figsize=(5, 5), facecolor='black')
plt.style.use('dark_background')
ax = fig.add_subplot(111, xlim=(-border, border), ylim=(-border, border))
ax.set_aspect('equal')

traces, = plt.plot([], [], 'y--', linewidth=1)
tracee, = plt.plot([], [], 'b--', linewidth=1)
tracem, = plt.plot([], [], 'r--', linewidth=1)

###################
# 滑条设置
ms_slider_ax = fig.add_axes([0.75, 0.3, 0.2, 0.02])
ms_slider = Slider(ms_slider_ax, 'ms', valmin=0, valmax=10, valinit=ms0)
ms_slider.label.set_size(10)
ms_slider.label.set_color('yellow')
xs_slider_ax = fig.add_axes([0.125, 0.0, 0.3, 0.02])
xs_slider = Slider(xs_slider_ax, 'xs', valmin=-100, valmax=100, valinit=xs0)
xs_slider.label.set_size(10)
xs_slider.label.set_color('yellow')
ys_slider_ax = fig.add_axes([0.5, 0.0, 0.3, 0.02])
ys_slider = Slider(ys_slider_ax, 'ys', valmin=-100, valmax=100, valinit=ys0)
ys_slider.label.set_size(10)
ys_slider.label.set_color('yellow')
vxs_slider_ax = fig.add_axes([0.125, 1 - 0.03, 0.3, 0.02])
vxs_slider = Slider(vxs_slider_ax, 'vxs', valmin=-1000, valmax=1000, valinit=vxs0)
vxs_slider.label.set_size(10)
vxs_slider.label.set_color('yellow')
vys_slider_ax = fig.add_axes([0.5, 1 - 0.03, 0.3, 0.02])
vys_slider = Slider(vys_slider_ax, 'vys', valmin=-1000, valmax=1000, valinit=vys0)
vys_slider.label.set_size(10)
vys_slider.label.set_color('yellow')

me_slider_ax = fig.add_axes([0.75, 0.33, 0.2, 0.02])
me_slider = Slider(me_slider_ax, 'me', valmin=0, valmax=10, valinit=me0)
me_slider.label.set_size(10)
me_slider.label.set_color('blue')
xe_slider_ax = fig.add_axes([0.125, 0.03, 0.3, 0.02])
xe_slider = Slider(xe_slider_ax, 'xe', valmin=-100, valmax=100, valinit=xe0)
xe_slider.label.set_size(10)
xe_slider.label.set_color('cyan')
ye_slider_ax = fig.add_axes([0.5, 0.03, 0.3, 0.02])
ye_slider = Slider(ye_slider_ax, 'ye', valmin=-100, valmax=100, valinit=ye0)
ye_slider.label.set_size(10)
ye_slider.label.set_color('cyan')
vxe_slider_ax = fig.add_axes([0.125, 1 - 0.06, 0.3, 0.02])
vxe_slider = Slider(vxe_slider_ax, 'vxe', valmin=-1000, valmax=1000, valinit=vxe0)
vxe_slider.label.set_size(10)
vxe_slider.label.set_color('cyan')
vye_slider_ax = fig.add_axes([0.5, 1 - 0.06, 0.3, 0.02])
vye_slider = Slider(vye_slider_ax, 'vye', valmin=-1000, valmax=1000, valinit=vye0)
vye_slider.label.set_size(10)
vye_slider.label.set_color('cyan')

mm_slider_ax = fig.add_axes([0.75, 0.36, 0.2, 0.02])
mm_slider = Slider(mm_slider_ax, 'mm', valmin=0, valmax=10, valinit=mm0)
mm_slider.label.set_size(10)
mm_slider.label.set_color('red')
xm_slider_ax = fig.add_axes([0.125, 0.06, 0.3, 0.02])
xm_slider = Slider(xm_slider_ax, 'xm', valmin=-100, valmax=100, valinit=xm0)
xm_slider.label.set_size(10)
xm_slider.label.set_color('red')
ym_slider_ax = fig.add_axes([0.5, 0.06, 0.3, 0.02])
ym_slider = Slider(ym_slider_ax, 'ym', valmin=-100, valmax=100, valinit=ym0)
ym_slider.label.set_size(10)
ym_slider.label.set_color('red')
vxm_slider_ax = fig.add_axes([0.125, 1 - 0.09, 0.3, 0.02])
vxm_slider = Slider(vxm_slider_ax, 'vxm', valmin=-1000, valmax=1000, valinit=vxm0)
vxm_slider.label.set_size(10)
vxm_slider.label.set_color('red')
vym_slider_ax = fig.add_axes([0.5, 1 - 0.09, 0.3, 0.02])
vym_slider = Slider(vym_slider_ax, 'vym', valmin=-1000, valmax=1000, valinit=vym0)
vym_slider.label.set_size(10)
vym_slider.label.set_color('red')
####################


xxs = [[xs_slider.val]]
yys = [[ys_slider.val]]
xxe = [[xe_slider.val]]
yye = [[ye_slider.val]]
xxm = [[xm_slider.val]]
yym = [[ym_slider.val]]


def animate(j):

    global xs, xe, xm, ys, ye, ym
    # 从滑条处得到初始位置、速度和质量条件
    xs0 = xs_slider.val
    xe0 = xe_slider.val
    xm0 = xm_slider.val
    ys0 = ys_slider.val
    ye0 = ye_slider.val
    ym0 = ym_slider.val
    vxs0 = vxs_slider.val
    vxe0 = vxe_slider.val
    vxm0 = vxm_slider.val
    vys0 = vys_slider.val
    vye0 = vye_slider.val
    vym0 = vym_slider.val
    ms = ms_slider.val
    me = me_slider.val
    mm = mm_slider.val

    xs, xe, xm, ys, ye, ym = Euler(ms, me, mm, xs0, xe0, xm0,vxs0, vxe0, vxm0, ys0, ye0, ym0, vys0, vye0, vym0)

    xxs.append(xs)
    yys.append(ys)
    xxe.append(xe)
    yye.append(ye)
    xxm.append(xm)
    yym.append(ym)

    xxs[j].append(xxs[j][-1])
    yys[j].append(yys[j][-1])
    xxe[j].append(xxe[j][-1])
    yye[j].append(yye[j][-1])
    xxm[j].append(xxm[j][-1])
    yym[j].append(yym[j][-1])

    traces.set_data(xxs[j], yys[j])
    tracee.set_data(xxe[j], yye[j])
    tracem.set_data(xxm[j], yym[j])

    return traces, tracee, tracem



anim = animation.FuncAnimation(fig, animate, frames=None, interval=1)
plt.show()

fig = plt.figure(figsize=(5, 5), facecolor='black')
plt.style.use('dark_background')
plt.plot(xs, ys, 'y--', label='Sun', linewidth=1)
plt.plot(xe, ye, 'b--', label='Earth', linewidth=1)
plt.plot(xm, ym, 'r--', label='Mars', linewidth=1)
plt.legend()
plt.show()

fig = plt.figure(figsize=(5, 5), facecolor='black')
plt.style.use('dark_background')
ax = fig.add_subplot(111, xlim=(-border, border), ylim=(-border, border))
ax.set_aspect('equal')

lines, = plt.plot([], [], 'yo')
linee, = plt.plot([], [], 'bo')
linem, = plt.plot([], [], 'ro')
traces, = plt.plot([], [], 'y--', linewidth=1)
tracee, = plt.plot([], [], 'b--', linewidth=1)
tracem, = plt.plot([], [], 'r--', linewidth=1)


def animate2(i):
    i *= speed
    lines.set_data(xs[i], ys[i])
    linee.set_data(xe[i], ye[i])
    linem.set_data(xm[i], ym[i])
    traces.set_data(xs[:i], ys[:i])
    tracee.set_data(xe[:i], ye[:i])
    tracem.set_data(xm[:i], ym[:i])

    return linee, linem, lines, traces, tracee, tracem


anim = animation.FuncAnimation(fig, animate2, frames=int(len(xs) / speed), interval=1, repeat=True)
plt.show()
