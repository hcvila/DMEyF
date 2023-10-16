# para correr el Google Cloud
#   8 vCPU
#  64 GB memoria RAM


# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

require("data.table")
require("lightgbm")


# defino los parametros de la corrida, en una lista, la variable global  PARAM
#  muy pronto esto se leera desde un archivo formato .yaml
PARAM <- list()
PARAM$experimento <- "KA8240_4"

PARAM$input$dataset <- "./datasets/competencia_02.csv"

# meses donde se entrena el modelo
PARAM$input$training <- c(201907, 201908, 201909, 201910, 201911, 201912, 
                          202011, 202012, 202101, 202102, 202103, 202104, 202105)
PARAM$input$future <- c(202107) # meses donde se aplica el modelo

PARAM$finalmodel$semilla <- 594421

# hiperparametros intencionalmente NO optimos
PARAM$finalmodel$optim$num_iterations <- 383
PARAM$finalmodel$optim$learning_rate <- 0.14088365
PARAM$finalmodel$optim$feature_fraction <- 0.52532600
PARAM$finalmodel$optim$min_data_in_leaf <- 9116
PARAM$finalmodel$optim$num_leaves <- 77


# Hiperparametros FIJOS de  lightgbm
PARAM$finalmodel$lgb_basicos <- list(
  boosting = "gbdt", # puede ir  dart  , ni pruebe random_forest
  objective = "binary",
  metric = "custom",
  first_metric_only = TRUE,
  boost_from_average = TRUE,
  feature_pre_filter = FALSE,
  force_row_wise = TRUE, # para reducir warnings
  verbosity = -100,
  max_depth = -1L, # -1 significa no limitar,  por ahora lo dejo fijo
  min_gain_to_split = 0.0, # min_gain_to_split >= 0.0
  min_sum_hessian_in_leaf = 0.001, #  min_sum_hessian_in_leaf >= 0.0
  lambda_l1 = 0.0, # lambda_l1 >= 0.0
  lambda_l2 = 0.0, # lambda_l2 >= 0.0
  max_bin = 31L, # lo debo dejar fijo, no participa de la BO
  
  bagging_fraction = 1.0, # 0.0 < bagging_fraction <= 1.0
  pos_bagging_fraction = 1.0, # 0.0 < pos_bagging_fraction <= 1.0
  neg_bagging_fraction = 1.0, # 0.0 < neg_bagging_fraction <= 1.0
  is_unbalance = FALSE, #
  scale_pos_weight = 1.0, # scale_pos_weight > 0.0
  
  drop_rate = 0.1, # 0.0 < neg_bagging_fraction <= 1.0
  max_drop = 50, # <=0 means no limit
  skip_drop = 0.5, # 0.0 <= skip_drop <= 1.0
  
  extra_trees = TRUE, # Magic Sauce
  
  seed = PARAM$finalmodel$semilla
)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa
setwd("~/buckets/b1")

# cargo el dataset donde voy a entrenar
dataset <- fread(PARAM$input$dataset, stringsAsFactors = TRUE)




# Catastrophe Analysis  -------------------------------------------------------
# deben ir cosas de este estilo
dataset[foto_mes == 202006, active_quarter := NA]
dataset[foto_mes %in% c(201905, 201910, 202006), mrentabilidad := NA]
dataset[foto_mes %in% c(201905, 201910, 202006), mrentabilidad_annual := NA]
dataset[foto_mes %in% c(201905, 201910, 202006), mcomisiones := NA]
dataset[foto_mes %in% c(201905, 201910, 202006), mactivos_margen := NA]
dataset[foto_mes %in% c(201905, 201910, 202006), mpasivos_margen := NA]
dataset[foto_mes == 202006, mcuentas_saldo := NA]
dataset[foto_mes == 202006, ctarjeta_debito_transacciones := NA]
dataset[foto_mes == 202006, mautoservicio := NA]
dataset[foto_mes == 202006, ctarjeta_visa_transacciones := NA]
dataset[foto_mes == 202006, mtarjeta_visa_consumo := NA]
dataset[foto_mes == 202006, ctarjeta_master_transacciones := NA]
dataset[foto_mes == 202006, mtarjeta_master_consumo := NA]
dataset[foto_mes == 202006, cprestamos_prendarios := NA]
dataset[foto_mes == 202006, mprestamos_prendarios := NA]
dataset[foto_mes == 202006, cprestamos_hipotecarios := NA]
dataset[foto_mes == 202006, mprestamos_hipotecarios := NA]
dataset[foto_mes %in% c(201905, 201910, 202006), ccomisiones_otras := NA]
dataset[foto_mes %in% c(201905, 201910, 202006), mcomisiones_otras := NA]
dataset[foto_mes ==201905, ctransferencias_recibidas := NA]
dataset[foto_mes ==201905, mtransferencias_recibidas := NA]
dataset[foto_mes == 202006, cextraccion_autoservicio := NA]
dataset[foto_mes == 202006, mextraccion_autoservicio := NA]
dataset[foto_mes == 202006, ccheques_depositados := NA]
dataset[foto_mes == 202006, mcheques_depositados := NA]
dataset[foto_mes == 202006, ccheques_emitidos := NA]
dataset[foto_mes == 202006, mcheques_emitidos := NA]
dataset[foto_mes == 202006, ccheques_depositados_rechazados := NA]
dataset[foto_mes == 202006, mcheques_depositados_rechazados := NA]
dataset[foto_mes == 202006, ccheques_emitidos_rechazados := NA]
dataset[foto_mes == 202006, mcheques_emitidos_rechazados := NA]
dataset[foto_mes == 202006, tcallcenter := NA]
dataset[foto_mes == 202006, ccallcenter_transacciones := NA]
dataset[foto_mes == 202006, thomebanking := NA]
dataset[foto_mes %in% c(201910, 202006), chomebanking_transacciones := NA]
dataset[foto_mes == 202006, ccajas_transacciones := NA]
dataset[foto_mes == 202006, ccajas_consultas := NA]
dataset[foto_mes == 202006, ccajas_depositos := NA]
dataset[foto_mes == 202006, ccajas_extracciones := NA]
dataset[foto_mes == 202006, ccajas_otras := NA]
dataset[foto_mes == 202006, catm_trx := NA]
dataset[foto_mes == 202006, matm := NA]
dataset[foto_mes == 202006, catm_trx_other := NA]
dataset[foto_mes == 202006, matm_other := NA]
dataset[foto_mes == 202006, ctrx_quarter := NA]
dataset[foto_mes == 202006, tmobile_app := NA]
dataset[foto_mes == 202006, cmobile_app_trx := NA]


# Feature Engineering Historico  ----------------------------------------------
#   aqui deben calcularse los  lags y  lag_delta

# los campos que se van a utilizar
campos <- setdiff(colnames(dataset), c("clase_ternaria", "foto_mes", "clase_ternaria"))

# lags hasta 6

for (i in 1:6){
  lagcolumns <- paste("lag",i,campos, sep="_")
  dataset[, (lagcolumns):= shift(.SD, type = "lag", fill = NA, n=i), .SDcols = campos,  by =numero_de_cliente]
}

# delta lag 

for (col in campos) {
  deltacolumns <- paste("delta1", col, sep="-")
  dataset[, (deltacolumns) := .SD[[col]] - .SD[[paste("lag_1", col, sep="_")]], .SDcols = c(col, paste("lag_1", col, sep="_"))]
}

for (col in campos) {
  deltacolumns2 <- paste("delta2", col, sep="-")
  dataset[, (deltacolumns2) := .SD[[col]] - .SD[[paste("lag_2", col, sep="_")]], .SDcols = c(col, paste("lag_2", col, sep="_"))]
}

for (col in campos) {
  deltacolumns3 <- paste("delta3", col, sep="-")
  dataset[, (deltacolumns3) := .SD[[col]] - .SD[[paste("lag_3", col, sep="_")]], .SDcols = c(col, paste("lag_3", col, sep="_"))]
}
for (col in campos) {
  deltacolumns4 <- paste("delta4", col, sep="-")
  dataset[, (deltacolumns4) := .SD[[col]] - .SD[[paste("lag_4", col, sep="_")]], .SDcols = c(col, paste("lag_4", col, sep="_"))]
}


#--------------------------------------

# paso la clase a binaria que tome valores {0,1}  enteros
# set trabaja con la clase  POS = { BAJA+1, BAJA+2 }
# esta estrategia es MUY importante
dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+2", "BAJA+1"), 1L, 0L)]

#--------------------------------------

# los campos que se van a utilizar
campos_buenos <- setdiff(colnames(dataset), c("clase_ternaria", "clase01"))

#--------------------------------------


# establezco donde entreno
dataset[, train := 0L]
dataset[foto_mes %in% PARAM$input$training, train := 1L]

#--------------------------------------
# creo las carpetas donde van los resultados
# creo la carpeta donde va el experimento
dir.create("./exp/", showWarnings = FALSE)
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory DEL EXPERIMENTO
setwd(paste0("./exp/", PARAM$experimento, "/"))



# dejo los datos en el formato que necesita LightGBM
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[train == 1L, campos_buenos, with = FALSE]),
  label = dataset[train == 1L, clase01]
)


# genero el modelo
param_completo <- c(PARAM$finalmodel$lgb_basicos,
                    PARAM$finalmodel$optim)

modelo <- lgb.train(
  data = dtrain,
  param = param_completo,
)

#--------------------------------------
# ahora imprimo la importancia de variables
tb_importancia <- as.data.table(lgb.importance(modelo))
archivo_importancia <- "impo.txt"

fwrite(tb_importancia,
       file = archivo_importancia,
       sep = "\t"
)

#--------------------------------------


# aplico el modelo a los datos sin clase
dapply <- dataset[foto_mes == PARAM$input$future]

# aplico el modelo a los datos nuevos
prediccion <- predict(
  modelo,
  data.matrix(dapply[, campos_buenos, with = FALSE])
)

# genero la tabla de entrega
tb_entrega <- dapply[, list(numero_de_cliente, foto_mes)]
tb_entrega[, prob := prediccion]

# grabo las probabilidad del modelo
fwrite(tb_entrega,
       file = "prediccion.txt",
       sep = "\t"
)

# ordeno por probabilidad descendente
setorder(tb_entrega, -prob)


# genero archivos con los  "envios" mejores
# deben subirse "inteligentemente" a Kaggle para no malgastar submits
# si la palabra inteligentemente no le significa nada aun
# suba TODOS los archivos a Kaggle
# espera a la siguiente clase sincronica en donde el tema sera explicado

cortes <- seq(8000, 15000, by = 500)
for (envios in cortes) {
  tb_entrega[, Predicted := 0L]
  tb_entrega[1:envios, Predicted := 1L]
  
  fwrite(tb_entrega[, list(numero_de_cliente, Predicted)],
         file = paste0(PARAM$experimento, "_", envios, ".csv"),
         sep = ","
  )
}

cat("\n\nLa generacion de los archivos para Kaggle ha terminado\n")
