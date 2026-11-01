"""Build the computational solution notebook for IMEC4001 Tarea 8."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "soluciones" / "LuisARodriguezA_tarea8.ipynb"
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


md(r"""
# Tarea 8 — Sistemas dinámicos y aparición de vórtices

**IMEC4001 · Solución computacional de estudio**  
**Entrega indicada en el enunciado: 1 de noviembre**

La primera parte clasifica y dibuja los equilibrios de un sistema plano para los cuatro valores indicados y para $$c=1$$, que se pide analizar aparte en el inciso f. La herramienta de estabilidad será el **diagrama de Poincaré**: cada equilibrio se representa mediante $$\operatorname{tr}A$$ y $$\det A$$ del Jacobiano. La segunda parte suma un vórtice de núcleo Rankine y un flujo uniforme, localiza los puntos de estancamiento, clasifica los que son hiperbólicos y representa las líneas de corriente.

> Las convenciones de flechas y los ejes de los retratos de fase se indican en cada figura. En las gráficas de velocidad se usa integración numérica para visualizar el campo; las posiciones y clasificaciones se obtienen analíticamente.
""")

md(r"""
## 1. Sistema no lineal expuesto en clase

El sistema del enunciado es

$$
\dot{x}=F_1(x,y)=x(y-y_0),\qquad
\dot{y}=F_2(x,y)=-y\left[(x-x_0)+c(y-y_0)\right],
$$

con $$x_0=y_0=1$$ y

$$c\in\left\{-\frac12,\;0,\;\frac12,\;2\right\}.$$

El inciso f pide además estudiar $$c=1$$, que también se incluye en la tabla y los retratos.

Los equilibrios satisfacen $$F_1=F_2=0$$. De $$x(y-1)=0$$ salen dos ramas:

1. Si $$y=1$$, la segunda ecuación exige $$x=1$$; siempre aparece $$P_*=(1,1)$$.
2. Si $$x=0$$, entonces $$-y[-1+c(y-1)]=0$$. Aparecen $$P_0=(0,0)$$ y, para $$c\ne0$$, $$P_c=(0,1+1/c)$$.

En los cuatro casos pedidos, $$P_c$$ coincide con $$P_0$$ cuando $$c=-1$$, valor que no está en la lista; por tanto no hay colisión de equilibrios en los casos estudiados.
""")

md(r"""
### Jacobiano y coordenadas de Poincaré

La matriz de linealización es

$$
A(x,y)=\begin{pmatrix}
y-1 & x\\
-y & -(x-1)-c(2y-1)
\end{pmatrix}.
$$

Para un equilibrio, definimos

$$\tau=\operatorname{tr}A,\qquad \delta=\det A,\qquad D=\tau^2-4\delta.$$

El diagrama de Poincaré se lee así:

| Región | Clasificación lineal |
|:--|:--|
| $$\delta<0$$ | silla |
| $$\delta>0,\ D>0,\ \tau<0$$ | nodo sumidero |
| $$\delta>0,\ D>0,\ \tau>0$$ | nodo fuente |
| $$\delta>0,\ D<0,\ \tau<0$$ | espiral sumidero |
| $$\delta>0,\ D<0,\ \tau>0$$ | espiral fuente |
| $$\delta>0,\ D<0,\ \tau=0$$ | centro lineal |
| $$D=0,\ \delta>0$$ | nodo repetido (posiblemente degenerado) |

La curva $$D=0$$ es $$\delta=\tau^2/4$$. El código dibuja esta frontera y ubica cada equilibrio según sus coordenadas $$(\tau,\delta)$$. En los casos no hiperbólicos (centro y autovalor repetido) se completa la conclusión con el sistema no lineal y los autovectores.
""")

code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 12, "axes.labelsize": 11})
cs = [-0.5, 0.0, 0.5, 2.0]
cs_todos = cs + [1.0]  # el inciso f pide también c=1

def campo(x, y, c):
    return x * (y - 1), -y * ((x - 1) + c * (y - 1))

def jacobiano(x, y, c):
    return np.array([[y - 1, x], [-y, -(x - 1) - c * (2*y - 1)]], dtype=float)

def equilibrios(c):
    pts = [(0.0, 0.0, r"$P_0=(0,0)$"), (1.0, 1.0, r"$P_*=(1,1)$")]
    if c != 0:
        yc = 1.0 + 1.0 / c
        if abs(yc) > 1e-10:  # evitar duplicar el origen en c=-1
            pts.append((0.0, yc, rf"$P_c=(0,{yc:g})$"))
    return pts

print("Equilibrios por parámetro:")
for c in cs_todos:
    print(f"c={c:g}: " + ", ".join(label for _, _, label in equilibrios(c)))
""")

md(r"""
### 1.a–b. Puntos de equilibrio y clasificación

En $$P_0=(0,0)$$,

$$A(P_0)=\begin{pmatrix}-1&0\\0&1+c\end{pmatrix},\quad
\tau=c,\quad\delta=-(1+c).$$

En $$P_*=(1,1)$$,

$$A(P_*)=\begin{pmatrix}0&1\\-1&-c\end{pmatrix},\quad
\tau=-c,\quad\delta=1,\quad D=c^2-4.$$

Para $$c\ne0$$, el tercer equilibrio es $$P_c=(0,1+1/c)$$ y

$$A(P_c)=\begin{pmatrix}1/c&0\\-(1+1/c)&-(1+c)\end{pmatrix},\quad
\lambda_1=1/c,\quad\lambda_2=-(1+c).$$

El siguiente bloque calcula estas cantidades directamente desde el Jacobiano y genera la tabla de clasificación.
""")

code(r"""
def clasifica(A):
    vals = np.linalg.eigvals(A)
    tau = float(np.trace(A))
    delta = float(np.linalg.det(A))
    disc = tau*tau - 4*delta
    tol = 1e-9
    if delta < -tol:
        clase = "silla"
    elif abs(disc) <= tol and delta > tol:
        clase = "nodo degenerado" if np.linalg.matrix_rank(A - np.eye(2)*tau/2) == 1 else "nodo estrella"
        clase += " sumidero" if tau < 0 else " fuente"
    elif disc > tol and delta > 0:
        clase = ("nodo sumidero" if tau < 0 else "nodo fuente")
    elif disc < -tol and delta > 0:
        if abs(tau) <= tol:
            clase = "centro lineal (revisar no linealidad)"
        else:
            clase = "espiral sumidero" if tau < 0 else "espiral fuente"
    else:
        clase = "no concluyente por linealización"
    return tau, delta, disc, vals, clase

rows = []
for c in cs_todos:
    for xeq, yeq, label in equilibrios(c):
        A = jacobiano(xeq, yeq, c)
        tau, delta, disc, vals, clase = clasifica(A)
        rows.append({"c": c, "Equilibrio": label, "tr A": tau, "det A": delta,
                     "D=(tr A)^2-4 det A": disc,
                     "Autovalores": ", ".join(f"{z.real:.4g}{z.imag:+.4g}i" for z in vals),
                     "Clasificación": clase})
tabla = pd.DataFrame(rows)
display(tabla.round(4))
""")

md(r"""
### Diagrama de Poincaré de los equilibrios

La abscisa representa $$\tau=\operatorname{tr}A$$ y la ordenada $$\delta=\det A$$. La parábola $$\delta=\tau^2/4$$ separa nodos de espirales; el semiplano $$\delta<0$$ contiene las sillas. El color indica el parámetro y la forma distingue el equilibrio.
""")

code(r"""
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
tt = np.linspace(-3.2, 3.2, 500)
ax.plot(tt, tt**2/4, color="black", lw=1.5, label=r"$D=0:\ \det A=(\operatorname{tr}A)^2/4$")
ax.axhline(0, color="0.35", lw=0.8)
ax.axvline(0, color="0.35", lw=0.8)
colors = {-0.5:"#d1495b", 0.0:"#00798c", 0.5:"#edae49", 2.0:"#30638e", 1.0:"#6a4c93"}
markers = {r"$P_0=(0,0)$":"o", r"$P_*=(1,1)$":"s"}
for c in cs_todos:
    for xeq, yeq, label in equilibrios(c):
        A = jacobiano(xeq, yeq, c)
        tau, delta, disc, vals, clase = clasifica(A)
        marker = "^" if label.startswith(r"$P_c") else markers[label]
        ax.scatter(tau, delta, s=90, color=colors[c], marker=marker, edgecolor="black",
                   linewidth=0.6, label=rf"$c={c:g}$, {label}")
        ax.annotate(rf"$c={c:g}$", (tau, delta), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.8, 3.0)
ax.set_xlabel(r"$\tau=\operatorname{tr}A$"); ax.set_ylabel(r"$\delta=\det A$")
ax.set_title("Diagrama de Poincaré: clasificación local de los equilibrios")
ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2, loc="lower right")
plt.show()
""")

md(r"""
### Casos que requieren interpretar la frontera

**En $$(1,1)$$ y $$c=0$$**, $$A^2=-I$$ y la linealización da un centro. Para confirmar el retrato no lineal, en el cuadrante $$x,y>0$$ se conserva

$$H(x,y)=x-\ln x+y-\ln y.$$

En efecto, $$dH/dt=0$$ cuando $$c=0$$; las curvas de nivel cerradas alrededor de $$(1,1)$$ son órbitas periódicas. Por tanto, es un **centro no lineal**, estable en el sentido de Lyapunov pero no asintóticamente estable.

**En $$(1,1)$$ y $$c=2$$**, $$D=0$$ y el autovalor repetido es $$-1$$. Hay un solo autovector independiente, así que es un **nodo degenerado sumidero**. La aproximación lineal domina localmente porque el autovalor no es cero.

En $$P_c$$ para $$c=-1/2$$ los dos autovalores son negativos: nodo sumidero. Para $$c=1/2$$ y $$c=2$$ tienen signos opuestos: silla.
""")

md(r"""
### 1.c–e. Direcciones principales y retratos de fase

Para cada equilibrio se calculan los autovectores del Jacobiano. En una silla, la dirección del autovector con autovalor negativo es estable y la del positivo es inestable. En nodos, las trayectorias se alinean con las direcciones propias; en espirales, el signo de $$\operatorname{tr}A$$ indica si entran o salen. Para el centro $$c=0$$, las curvas cerradas siguen la integral primera $$H$$.

Las gráficas muestran líneas de corriente y flechas del campo. Las líneas de corriente describen la geometría del campo autónomo; su parametrización no equivale necesariamente a un muestreo uniforme en tiempo.
""")

code(r"""
def retrato(c, lim=(-2.2, 3.2, -2.0, 4.2), n=31):
    xmin, xmax, ymin, ymax = lim
    xx = np.linspace(xmin, xmax, n); yy = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(xx, yy)
    U, V = campo(X, Y, c)
    speed = np.hypot(U, V)
    fig, ax = plt.subplots(figsize=(7.3, 6.0), constrained_layout=True)
    ax.streamplot(X, Y, U, V, density=1.15, color=np.log1p(speed), cmap="viridis",
                  linewidth=0.9, arrowsize=1.0, maxlength=4.0)
    ax.quiver(X[::3,::3], Y[::3,::3], U[::3,::3], V[::3,::3], color="0.25",
              alpha=0.55, pivot="mid", scale=45)
    for xe, ye, label in equilibrios(c):
        ax.scatter(xe, ye, s=65, c="#e63946", edgecolor="white", linewidth=0.9, zorder=5)
        ax.annotate(label, (xe, ye), xytext=(6, 7), textcoords="offset points", fontsize=9)
        A = jacobiano(xe, ye, c)
        vals, vecs = np.linalg.eig(A)
        span = 0.55
        for lam, vec in zip(vals, vecs.T):
            if abs(lam.imag) < 1e-8:
                v = np.real(vec); v = v / np.linalg.norm(v)
                for sign in (-1, 1):
                    ax.plot([xe, xe+sign*span*v[0]], [ye, ye+sign*span*v[1]],
                            color="#f4a261" if lam.real < 0 else "#d00000",
                            lw=1.6, ls="--" if lam.real < 0 else ":", zorder=4)
    ax.set(xlim=(xmin,xmax), ylim=(ymin,ymax), xlabel="$x$", ylabel="$y$",
           title=rf"Retrato de fase del sistema, $c={c:g}$")
    ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="black", lw=0.6)
    ax.grid(alpha=0.2)
    ax.plot([], [], color="#f4a261", ls="--", label="dirección propia estable")
    ax.plot([], [], color="#d00000", ls=":", label="dirección propia inestable")
    ax.legend(loc="upper right", fontsize=8)
    plt.show()

for c in cs_todos:
    retrato(c)
""")

md(r"""
### Interpretación de la primera parte

| $$c$$ | $$P_0=(0,0)$$ | $$P_*=(1,1)$$ | $$P_c=(0,1+1/c)$$ |
|--:|:--|:--|:--|
| $$-1/2$$ | silla | espiral fuente | nodo sumidero en $$(0,-1)$$ |
| $$0$$ | silla | centro no lineal | no existe equilibrio adicional |
| $$1/2$$ | silla | espiral sumidero | silla en $$(0,3)$$ |
| $$2$$ | silla | nodo degenerado sumidero | silla en $$(0,3/2)$$ |
| $$1$$ (inciso f) | silla | espiral sumidero | silla en $$(0,2)$$ |

El origen siempre es una silla para estos parámetros: sus autovalores son $$-1$$ y $$1+c$$, de signos opuestos. En $$P_*$$, la transición de fuente a centro y luego a sumidero se aprecia en el diagrama $(\tau,\delta)$.
""")

md(r"""
## 2. Aparición de vórtices en un flujo

El flujo inducido por un vórtice con núcleo rotacional tiene velocidad tangencial

$$
V_\theta(r)=\begin{cases}
\omega r,&r\le r_0,\\[1mm]
\dfrac{\omega r_0^2}{r},&r\ge r_0.
\end{cases}
$$

En coordenadas cartesianas, si $$r=\sqrt{x^2+y^2}$$,

$$\mathbf V_{\rm vort}(x,y)=\begin{cases}
(-\omega y,\;\omega x),&r\le r_0,\\[1mm]
\left(-\dfrac{\omega r_0^2y}{r^2},\;\dfrac{\omega r_0^2x}{r^2}\right),&r\ge r_0.
\end{cases}$$

Se suma un flujo uniforme horizontal $$\mathbf V_\infty=(U,0)$$. Tomamos $$\omega>0$$ y $$U\ge0$$; los números reportados se expresan con $$\Omega=|\omega|$$. Para $$\omega<0$$, los puntos se reflejan respecto al eje $$x$$. La rapidez tangencial alcanza su máximo en $$r=r_0$$:

$$U_{\rm crit}=\Omega r_0.$$
""")

md(r"""
### 2.a. Localización y número de puntos de estancamiento

En un cero, la componente vertical del vórtice debe ser cero. Para $$r>0$$ esto requiere $$x=0$$, así que los ceros están en el eje vertical. Para $$\omega>0$$ y $$U>0$$, solo el semieje $$y>0$$ cancela la corriente uniforme. La ecuación escalar es

$$U=\begin{cases}
\Omega r,&0<r\le r_0,\\
\dfrac{\Omega r_0^2}{r},&r\ge r_0.
\end{cases}$$

Por tanto:

$$
\begin{array}{c|c|c}
\text{régimen}&\text{radio(s)}&\text{número de ceros}\n\\\hline
U>\Omega r_0&\text{ninguno}&0\\
U=\Omega r_0&r=r_0&1\\
0<U<\Omega r_0&r_{\rm int}=U/\Omega,\quad r_{\rm ext}=\Omega r_0^2/U&2
\end{array}
$$

Los dos radios del último renglón cumplen $$r_{\rm int}<r_0<r_{\rm ext}$$ y $$r_{\rm int}r_{\rm ext}=r_0^2$$. La igualdad se sigue al empalmar las dos ramas de rapidez en el borde del núcleo. Para $$U=0$$ existe además el cero en el centro $$r=0$$.
""")

code(r"""
def radios_criticos(U, omega=1.0, r0=1.0):
    Om = abs(omega)
    if U < 0 or Om == 0 or r0 <= 0:
        raise ValueError("Se requiere U >= 0, |omega| > 0 y r0 > 0.")
    Ucrit = Om*r0
    tol = 1e-10*max(1.0, Ucrit)
    if U == 0:
        return [(0.0, "centro")]
    if U > Ucrit + tol:
        return []
    if abs(U-Ucrit) <= tol:
        return [(r0, "borde")]
    return [(U/Om, "interior"), (Om*r0**2/U, "exterior")]

omega, r0 = 1.0, 1.0
Ucrit = omega*r0
casos_U = [(1.35*Ucrit, "sin ceros"), (Ucrit, "umbral"), (0.55*Ucrit, "dos ceros")]
for U, nombre in casos_U:
    print(f"{nombre:12s} U={U:.3f}: {radios_criticos(U,omega,r0)}")
""")

md(r"""
### 2.b. Clasificación de los ceros

El Jacobiano del vórtice dentro del núcleo es

$$A_{\rm int}=\begin{pmatrix}0&-\omega\\\omega&0\end{pmatrix},\qquad
\operatorname{tr}A_{\rm int}=0,\quad\det A_{\rm int}=\omega^2>0.$$

Sus autovalores son $$\lambda=\pm i\omega$$: el cero interior es un **centro lineal**. En el núcleo el campo total es exactamente afín, así que las trayectorias cercanas son círculos: el centro también queda confirmado no linealmente.

Fuera del núcleo, sobre el eje $$x=0,y=r>r_0$$,

$$A_{\rm ext}=\begin{pmatrix}0&q\\q&0\end{pmatrix},\qquad
q=\frac{\omega r_0^2}{r^2},\quad
\operatorname{tr}A_{\rm ext}=0,\quad\det A_{\rm ext}=-q^2<0.$$

Es una **silla**, con autovalores $$\lambda_+=q,\lambda_-=-q$$ y autovectores $$\mathbf v_+=(1,1)^T$$ (inestable) y $$\mathbf v_-=(1,-1)^T$$ (estable), para $$\omega>0$$. En los casos de dos ceros hay, por tanto, un centro interior y una silla exterior.

En el umbral $$U=\Omega r_0$$, el único cero está sobre la interfaz $$r=r_0$$. La velocidad es continua, pero su derivada radial salta: no existe un Jacobiano único. Las linealizaciones unilaterales son centro desde el núcleo y silla desde el exterior; el equilibrio del umbral es **no hiperbólico y no se clasifica con un solo retrato lineal**. Es el caso límite donde los dos ceros se unen en el borde.
""")

code(r"""
def velocidad_total(x, y, U, omega=1.0, r0=1.0):
    r2 = x*x + y*y
    r = np.sqrt(r2)
    coef = np.where(r <= r0, omega, omega*r0**2/np.maximum(r2, 1e-15))
    return U - coef*y, coef*x

print("Puntos y clasificación para omega=r0=1:")
for U, nombre in casos_U:
    print(f"\n{nombre}, U={U:.3f}")
    for r, region in radios_criticos(U,omega,r0):
        if region == "interior":
            print(f"  (0,{r:.4f}), interior: centro, autovalores ±i{omega:g}")
        elif region == "exterior":
            q = omega*r0**2/r**2
            print(f"  (0,{r:.4f}), exterior: silla, autovalores ±{q:.4f}")
        elif region == "borde":
            print(f"  (0,{r:.4f}), borde: equilibrio degenerado (Jacobiano unilateral)")
""")

md(r"""
### 2.c. Líneas de corriente para los tres regímenes

Para un flujo plano incompresible, una función de corriente satisface $$u=\partial_y\psi$$ y $$v=-\partial_x\psi$$. Para este campo,

$$\psi(x,y)=Uy+\begin{cases}
\dfrac{\omega}{2}(x^2+y^2),&r\le r_0,\\[1mm]
\omega r_0^2\left[\dfrac12+\ln(r/r_0)\right],&r\ge r_0.
\end{cases}$$

La constante exterior se eligió para que $$\psi$$ sea continua en $$r=r_0$$. Las líneas de nivel $$\psi=\text{constante}$$ son las líneas de corriente. El código colorea los niveles de $$\psi$$ y superpone flechas del campo; los puntos de estancamiento se marcan en rojo.
""")

code(r"""
def funcion_corriente(X, Y, U, omega=1.0, r0=1.0):
    R = np.hypot(X,Y)
    core = 0.5*omega*R**2
    outer = omega*r0**2*(0.5 + np.log(np.maximum(R/r0, 1e-12)))
    vort_psi = np.where(R <= r0, core, outer)
    return U*Y + vort_psi

def grafica_vortice(U, nombre, omega=1.0, r0=1.0, lim=3.2):
    x = np.linspace(-lim,lim,151); y = np.linspace(-lim,lim,151)
    X,Y = np.meshgrid(x,y)
    VX,VY = velocidad_total(X,Y,U,omega,r0)
    PSI = funcion_corriente(X,Y,U,omega,r0)
    fig,ax = plt.subplots(figsize=(7.1,6.2), constrained_layout=True)
    levels = np.linspace(np.nanpercentile(PSI,5), np.nanpercentile(PSI,95), 34)
    ax.contour(X,Y,PSI,levels=levels,cmap="viridis",linewidths=0.9)
    ax.streamplot(x,y,VX,VY,color="0.30",density=0.9,linewidth=0.55,arrowsize=0.8)
    ax.add_patch(plt.Circle((0,0),r0,fill=False,color="#e76f51",ls="--",lw=1.4,label="$r=r_0$"))
    pts = radios_criticos(U,omega,r0)
    for r, region in pts:
        if region == "centro":
            ax.scatter(0,0,s=70,c="#d00000",edgecolor="white",zorder=5)
        else:
            ax.scatter(0,r,s=70,c="#d00000",edgecolor="white",zorder=5)
            ax.annotate(rf"$r={r:.2f}$",(0,r),xytext=(8,6),textcoords="offset points",fontsize=8)
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect="equal",xlabel="$x$",ylabel="$y$",
           title=rf"{nombre}: $U={U:.2f}$, $U_{{crit}}={abs(omega)*r0:.2f}$")
    ax.axhline(0,color="0.4",lw=0.6); ax.axvline(0,color="0.4",lw=0.6)
    ax.legend(loc="lower right"); ax.grid(alpha=0.15)
    plt.show()

for U,nombre in casos_U:
    grafica_vortice(U,nombre,omega,r0)
""")

md(r"""
### Resumen de la segunda parte

| Condición | Puntos de estancamiento | Tipo |
|:--|:--|:--|
| $$U>\Omega r_0$$ | ninguno | el flujo uniforme domina |
| $$U=\Omega r_0$$ | uno en $$r=r_0$$ | umbral no hiperbólico; empalme de linealizaciones |
| $$0<U<\Omega r_0$$ | $$r_{\rm int}=U/\Omega$$ y $$r_{\rm ext}=\Omega r_0^2/U$$ | centro interior y silla exterior |
| $$U=0$$ | centro del núcleo $$r=0$$ | centro lineal |

La circulación del vórtice no produce siempre un par de puntos de estancamiento: aparecen dos cuando la corriente uniforme es menor que la rapidez máxima del vórtice en el borde del núcleo.
""")

md(r"""
## Guía breve para estudiar el examen

1. **Equilibrios:** factorizar cada componente del campo y resolver las ramas por separado.
2. **Estabilidad:** evaluar el Jacobiano en cada equilibrio; calcular $$\tau$$, $$\delta$$ y $$D=\tau^2-4\delta$$; ubicar el punto en el diagrama de Poincaré.
3. **Direcciones del retrato:** calcular autovectores. En sillas, identificar explícitamente las direcciones estable e inestable.
4. **Casos límite:** si $$\tau=0$$ o $$D=0$$, la clasificación lineal puede requerir información adicional del sistema no lineal.
5. **Vórtice + corriente:** localizar los ceros sobre el eje vertical, resolver por separado dentro y fuera del núcleo y revisar que cada radio caiga en la región usada.
6. **Dibujos:** verificar siempre que las flechas coincidan con el campo, no solo con la forma de las curvas.

Las gráficas de esta libreta ayudan a visualizar los resultados; para una respuesta de examen, conviene acompañar cada clasificación con los valores de traza, determinante y autovalores.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nbf.write(nb, OUTPUT)
print(f"Escrito: {OUTPUT}")
print(f"Celdas: {len(cells)}")
