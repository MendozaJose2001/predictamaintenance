import rpy2.robjects as ro

ro.r('''
library(frailtypack)
library(survival)

data(kidney)
fit <- frailtyPenal(
    Surv(time, status) ~ cluster(id) + sex + age,
    n.knots=7, kappa=1000, data=kidney
)

# Ver qué retorna predict
new_data <- data.frame(sex=c(1,2), age=c(30,50))
pred <- predict(fit, newdata=new_data, t=seq(1, 100, length=10))

cat("Nombres del objeto pred:\n")
print(names(pred))

cat("\nDimensiones de survival:\n")
print(dim(pred$survival))

cat("\nPrimeras filas de survival:\n")
print(head(pred$survival))

cat("\nTimes:\n")
print(pred$times)
''')