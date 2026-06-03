## Ecuaciones de Markowitz
Supongamos una cartera de dos activos con:

- rentabilidades esperadas $\mu_1$ y $\mu_2$
- varianzas $\sigma_1^2$ y $\sigma_2^2$
- correlación $\rho_{12}$
- pesos $w_1$ y $w_2$, donde $w_1 + w_2 = 1$

Las fórmulas principales de Markowitz son:

### Rentabilidad esperada de la cartera

$$
\mu_p = w_1 \mu_1 + w_2 \mu_2
$$

Como $w_2 = 1 - w_1$, también puede escribirse:
$$
\mu_p = w_1 \mu_1 + (1-w_1)\mu_2
$$
o:
$$
\mu_p = \mu_2 + w_1(\mu_1 - \mu_2)
$$

### Varianza de la cartera

$$
\sigma_p^2 =
w_1^2\sigma_1^2
+
w_2^2\sigma_2^2
+
2w_1w_2\rho_{12}\sigma_1\sigma_2
$$

Como $w_2 = 1-w_1$:
$$
\sigma_p^2 =
w_1^2\sigma_1^2
+
(1-w_1)^2\sigma_2^2
+
2w_1(1-w_1)\rho_{12}\sigma_1\sigma_2
$$

### Volatilidad de la cartera

$$
\sigma_p =
\sqrt{
w_1^2\sigma_1^2
+
w_2^2\sigma_2^2
+
2w_1w_2\rho_{12}\sigma_1\sigma_2
}
$$

o, sustituyendo $w_2 = 1-w_1$:
$$
\sigma_p =
\sqrt{
w_1^2\sigma_1^2
+
(1-w_1)^2\sigma_2^2
+
2w_1(1-w_1)\rho_{12}\sigma_1\sigma_2
}
$$
