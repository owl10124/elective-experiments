# Basic raytracing attempt

**Point:** honestly just a pair of pairs

**Tri:** a thing defined by three points

- Normal: $(P_2-P_1)\times (P_3-P_1)$
- Flip points to flip normal

**Ray:** origin, unit vector



Intercept(ray,vector):
$$
\begin{align}
\left(\begin{matrix}
x_0+\gamma x\\
y_0+\gamma y\\
z_0+\gamma z\\
\end{matrix}\right)
&= \left(\begin{matrix}
x_1+\alpha x_a+\beta x_b\\
y_1+\alpha y_a+\beta y_b\\
z_1+\alpha z_a+\beta z_b\\
\end{matrix}\right)
\\
\left(\begin{matrix}
x&x_a&x_b\\y&y_a&y_b\\z&z_a&z_b\\
\end{matrix}\right)
\left(\begin{matrix}
-\gamma\\\alpha\\\beta
\end{matrix}\right)
&=
\left(\begin{matrix}
x_0-x_1\\y_0-y_1\\z_0-z_1
\end{matrix}\right)
\end{align}
$$
return $\alpha, \beta$

and if $\alpha,\beta>0, \alpha+\beta<1$ then within

yay!

---

quaternions or sth

$\left|a+bi+cj+dk\right|=1$



projecc

it's just like ratio right

say you have $w,x,y,z$

then that becomes $\frac{x,y,z}{1+w}$