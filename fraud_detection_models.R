# ============================================
# AMS6104 Fraud Project - Part 2.1 Data Understanding &EDA
# ============================================

# Check if datasets exist
if (!exists("fraudTrain") || !exists("fraudTest")) {
  stop("❌ Error: fraudTrain or fraudTest not found. Please load the data first.")
}

# Use shorter variable names
train_data <- fraudTrain
test_data <- fraudTest

# Make column names lowercase for consistency
names(train_data) <- tolower(names(train_data))
names(test_data) <- tolower(names(test_data))

cat("=== 1) Dataset Structure ===\n")

# Basic dataset info
n_rows <- nrow(train_data)
n_cols <- ncol(train_data)

# Count numeric and categorical columns
numeric_cols <- sum(sapply(train_data, is.numeric))
categorical_cols <- sum(sapply(train_data, function(x) is.character(x) || is.factor(x)))

cat("Training dataset has", n_rows, "rows and", n_cols, "columns\n")
cat("Numeric columns:", numeric_cols, "| Categorical columns:", categorical_cols, "\n")

# Check important columns
important_cols <- c("amt", "city_pop", "gender", "state", "category", "trans_date_trans_time", "unix_time")
found_cols <- important_cols[important_cols %in% names(train_data)]
cat("Important columns found:", paste(found_cols, collapse = ", "), "\n")

cat("\n=== 2) Target Variable Analysis ===\n")

# Check if target variable exists
if (!"is_fraud" %in% names(train_data)) {
  stop("❌ Error: is_fraud column not found in training data")
}

# Convert target to factor with clear levels
convert_to_fraud_factor <- function(x) {
  # Handle different possible formats of fraud indicator
  fraud_indicator <- tolower(as.character(x)) %in% c("1", "yes", "y", "true", "t")
  factor(ifelse(fraud_indicator, "yes", "no"), levels = c("no", "yes"))
}

train_data$is_fraud <- convert_to_fraud_factor(train_data$is_fraud)
test_data$is_fraud <- convert_to_fraud_factor(test_data$is_fraud)

# Calculate fraud percentages
fraud_percent_train <- round(mean(train_data$is_fraud == "yes") * 100, 4)
fraud_percent_test <- round(mean(test_data$is_fraud == "yes") * 100, 4)

cat("Fraud rate in training data:", fraud_percent_train, "%\n")
cat("Fraud rate in test data:", fraud_percent_test, "%\n")

# Check if data is highly imbalanced
if (fraud_percent_train < 1) {
  cat("⚠️  Highly imbalanced dataset detected!\n")
  cat("💡 Recommendation: Use SMOTE for balancing and focus on AUC/Recall metrics\n")
}

cat("\n=== 3) Key Feature Exploration ===\n")

# Analyze transaction amount
if ("amt" %in% names(train_data)) {
  amt_data <- train_data$amt[is.finite(train_data$amt)]
  amt_stats <- summary(amt_data)
  cat("Transaction Amount (amt):\n")
  cat("  Min:", round(amt_stats["Min."], 2), "| Median:", round(amt_stats["Median"], 2), "\n")
  cat("  Mean:", round(amt_stats["Mean"], 2), "| Max:", round(amt_stats["Max."], 2), "\n")
  
  # Check if log transformation might be helpful
  if (amt_stats["Mean"] > amt_stats["Median"] * 2) {
    cat("  💡 Right-skewed: Consider using log(amt) for modeling\n")
  }
} else {
  cat("Transaction amount (amt) column not found\n")
}

# Analyze transaction time
if ("trans_date_trans_time" %in% names(train_data)) {
  # Extract hour from transaction time
  transaction_hours <- as.numeric(format(as.POSIXct(train_data$trans_date_trans_time), "%H"))
  hour_counts <- table(transaction_hours)
  top_hours <- names(sort(hour_counts, decreasing = TRUE))[1:3]
  cat("Transaction Hours:\n")
  cat("  Most frequent hours:", paste(top_hours, collapse = ", "), "\n")
  cat("  💡 Consider creating: hour, is_weekend, is_night features\n")
} else if ("unix_time" %in% names(train_data)) {
  cat("Unix time available - can extract hour information\n")
} else {
  cat("No transaction time columns found\n")
}

# Analyze city population
if ("city_pop" %in% names(train_data)) {
  pop_data <- train_data$city_pop[is.finite(train_data$city_pop)]
  pop_stats <- summary(pop_data)
  cat("City Population:\n")
  cat("  Min:", round(pop_stats["Min."], 0), "| Median:", round(pop_stats["Median"], 0), "\n")
  cat("  Max:", round(pop_stats["Max."], 0), "\n")
  
  if (pop_stats["Max."] > pop_stats["Median"] * 100) {
    cat("  💡 Large range: Consider using log(city_pop)\n")
  }
} else {
  cat("City population (city_pop) column not found\n")
}

cat("\n=== 4) Summary for Copy ===\n")
cat("Dataset:", n_rows, "rows,", n_cols, "columns\n")

# Calculate fraud to non-fraud ratio
non_fraud_count_train <- sum(train_data$is_fraud == "no")
fraud_count_train <- sum(train_data$is_fraud == "yes")
ratio_train <- round(non_fraud_count_train / fraud_count_train, 1)

non_fraud_count_test <- sum(test_data$is_fraud == "no")
fraud_count_test <- sum(test_data$is_fraud == "yes") 
ratio_test <- round(non_fraud_count_test / fraud_count_test, 1)

cat("Fraud rate: Train =", fraud_percent_train, "%, Test =", fraud_percent_test, "%\n")
cat("Class imbalance: 1 fraud in", ratio_train, "transactions (Train)\n")
cat("Class imbalance: 1 fraud in", ratio_test, "transactions (Test)\n")
cat("Data imbalance:", ifelse(fraud_percent_train < 1, "High (needs SMOTE)", "Moderate"), "\n")

# Show detailed information for key features
cat("\nKey Features Details:\n")

# Transaction amount details
if ("amt" %in% names(train_data)) {
  amt_data <- train_data$amt[is.finite(train_data$amt)]
  amt_stats <- summary(amt_data)
  amt_fraud_stats <- summary(train_data$amt[train_data$is_fraud == "yes"])
  amt_nonfraud_stats <- summary(train_data$amt[train_data$is_fraud == "no"])
  
  cat("- Transaction Amount (amt):\n")
  cat("  Overall: min=$", round(amt_stats["Min."], 2), 
      ", median=$", round(amt_stats["Median"], 2),
      ", mean=$", round(amt_stats["Mean"], 2), 
      ", max=$", round(amt_stats["Max."], 2), "\n", sep = "")
  cat("  Fraud transactions: median=$", round(amt_fraud_stats["Median"], 2),
      ", mean=$", round(amt_fraud_stats["Mean"], 2), "\n", sep = "")
  cat("  Legit transactions: median=$", round(amt_nonfraud_stats["Median"], 2),
      ", mean=$", round(amt_nonfraud_stats["Mean"], 2), "\n", sep = "")
}

# Transaction time details
if ("trans_date_trans_time" %in% names(train_data)) {
  transaction_hours <- as.numeric(format(as.POSIXct(train_data$trans_date_trans_time), "%H"))
  fraud_hours <- transaction_hours[train_data$is_fraud == "yes"]
  legit_hours <- transaction_hours[train_data$is_fraud == "no"]
  
  hour_counts <- table(transaction_hours)
  fraud_hour_counts <- table(fraud_hours)
  
  # Find top 3 hours for all transactions and for frauds
  top_hours_all <- names(sort(hour_counts, decreasing = TRUE))[1:3]
  top_hours_fraud <- names(sort(fraud_hour_counts, decreasing = TRUE))[1:min(3, length(fraud_hour_counts))]
  
  cat("- Transaction Time:\n")
  cat("  Most frequent hours (all):", paste(top_hours_all, collapse = ", "), "\n")
  cat("  Most frequent hours (fraud):", paste(top_hours_fraud, collapse = ", "), "\n")
  
  # Check if fraud pattern differs from overall pattern
  if (length(intersect(top_hours_all, top_hours_fraud)) == 0) {
    cat("  ⚠️  Fraud pattern differs from overall transaction pattern\n")
  }
}

cat("\n✅ Data understanding completed! Ready for preprocessing and SMOTE.\n")

# =========================================================
# AMS6104 Fraud Project — Part 2.2 DATA PROCESSING
# =========================================================

# ---------- 0) Packages ----------
req_pkgs <- c("data.table","dplyr","lubridate","geosphere",
              "forcats","recipes","tidyr","tidymodels","ROSE","Matrix","tools")
to_install <- setdiff(req_pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)
suppressPackageStartupMessages({
  library(data.table); library(dplyr); library(lubridate); library(geosphere)
  library(forcats); library(recipes); library(tidymodels); library(tidyr)
  library(Matrix); library(ROSE); library(tools)
})
set.seed(6104)

# ---------- Check if processed dataframes already exist in environment ----------
dataframes_exist <- exists("train_stratsample_smote") && exists("test_stratsample_50k_fe")

if (dataframes_exist) {
  cat("✅ Preprocessed dataframes found in current environment.\n")
  
  # Verify they are dataframes and have the expected structure
  if (is.data.frame(train_stratsample_smote) && is.data.frame(test_stratsample_50k_fe)) {
    
    # Convert target to factor with correct levels if needed
    if (!is.factor(train_stratsample_smote$is_fraud)) {
      train_stratsample_smote$is_fraud <- factor(train_stratsample_smote$is_fraud, levels = c("no", "yes"))
    }
    if (!is.factor(test_stratsample_50k_fe$is_fraud)) {
      test_stratsample_50k_fe$is_fraud <- factor(test_stratsample_50k_fe$is_fraud, levels = c("no", "yes"))
    }
    
    cat("📁 Using existing dataframes:\n")
    cat(sprintf("  - train_stratsample_smote: %d rows, %d columns\n", 
                nrow(train_stratsample_smote), ncol(train_stratsample_smote)))
    cat(sprintf("  - test_stratsample_50k_fe: %d rows, %d columns\n", 
                nrow(test_stratsample_50k_fe), ncol(test_stratsample_50k_fe)))
    
    # Calculate and display fraud rates
    train_fraud_rate <- mean(train_stratsample_smote$is_fraud == "yes") * 100
    test_fraud_rate <- mean(test_stratsample_50k_fe$is_fraud == "yes") * 100
    
    cat(sprintf("  - Train fraud rate: %.2f%%\n", train_fraud_rate))
    cat(sprintf("  - Test fraud rate:  %.2f%%\n", test_fraud_rate))
    
    # If train fraud rate is too low (indicating SMOTE wasn't applied), we should reprocess
    if (train_fraud_rate < 5) {
      cat("⚠️ Train fraud rate is too low (SMOTE may not have been applied). Forcing reprocessing...\n")
      dataframes_exist <- FALSE
    }
  } else {
    cat("⚠️ Objects exist but are not dataframes. Starting data processing pipeline...\n")
    dataframes_exist <- FALSE
  }
}

if (!dataframes_exist) {
  cat("🔧 Preprocessed data not found in environment. Starting data processing pipeline...\n")
  
  # ---------- Force reprocessing to ensure SMOTE is applied correctly ----------
  cat("🔄 Forcing data reprocessing to ensure SMOTE is applied correctly...\n")
  
  # ---------- Step 1) Load + upfront cleaning + stratified sampling ----------
  # Input from Environment
  if (!exists("fraudTrain") || !exists("fraudTest")) {
    stop("Objects 'fraudTrain' and/or 'fraudTest' not found in Environment.")
  }
  tr0 <- fraudTrain; te0 <- fraudTest
  
  # Standardize column names and remove exact duplicates
  names(tr0) <- tolower(names(tr0)); names(te0) <- tolower(names(te0))
  tr0 <- unique(tr0); te0 <- unique(te0)
  
  # Convert target to "yes"/"no" (as character first)
  if ("is_fraud" %in% names(tr0)) tr0$is_fraud <- ifelse(tr0$is_fraud == 1, "yes", "no")
  if ("is_fraud" %in% names(te0)) te0$is_fraud <- ifelse(te0$is_fraud == 1, "yes", "no")
  
  # Strict removal: drop rows with ANY NA across all columns
  tr0 <- tidyr::drop_na(tr0)
  te0 <- tidyr::drop_na(te0)
  
  # Finalize target as factor (required for stratified sampling and downstream)
  if ("is_fraud" %in% names(tr0)) tr0$is_fraud <- factor(tr0$is_fraud, levels = c("no","yes"))
  if ("is_fraud" %in% names(te0)) te0$is_fraud <- factor(te0$is_fraud, levels = c("no","yes"))
  
  cat(sprintf("After de-dup + drop NA — Train: %d rows | Test: %d rows\n", nrow(tr0), nrow(te0)))
  
  # Stratified sampling to fixed totals
  stratified_sample <- function(df, target_total, y = "is_fraud") {
    stopifnot(y %in% names(df))
    tab <- prop.table(table(df[[y]]))
    p_yes <- as.numeric(tab["yes"]); if (is.na(p_yes)) p_yes <- 0
    n_yes <- max(1L, round(target_total * p_yes))
    n_no  <- max(1L, target_total - n_yes)
    yes_pool <- dplyr::filter(df, .data[[y]] == "yes")
    no_pool  <- dplyr::filter(df, .data[[y]] == "no")
    dplyr::bind_rows(
      if (nrow(yes_pool) > n_yes) dplyr::slice_sample(yes_pool, n = n_yes) else yes_pool,
      if (nrow(no_pool)  > n_no)  dplyr::slice_sample(no_pool,  n = n_no)  else no_pool
    ) %>% dplyr::sample_frac(1.0)
  }
  
  # Sample (Train 20k / Test 50k)
  train_srs20k <- stratified_sample(tr0, 20000L)
  test_srs50k  <- stratified_sample(te0, 50000L)
  
  cat(sprintf("Sampled Train: %d rows (yes=%d, no=%d)\n",
              nrow(train_srs20k),
              sum(train_srs20k$is_fraud=="yes"),
              sum(train_srs20k$is_fraud=="no")))
  cat(sprintf("Sampled Test : %d rows (yes=%d, no=%d)\n",
              nrow(test_srs50k),
              sum(test_srs50k$is_fraud=="yes"),
              sum(test_srs50k$is_fraud=="no")))
  
  # ---------- Step 2) Common FE on sampled sets ----------
  # 2.1 Time parsing and calendar splits
  parse_time <- function(dt) {
    if ("trans_date_trans_time" %in% names(dt)) {
      dt %>% mutate(trans_dt = ymd_hms(as.character(trans_date_trans_time), quiet = TRUE))
    } else if ("unix_time" %in% names(dt)) {
      dt %>% mutate(trans_dt = as_datetime(unix_time))
    } else stop("Require either trans_date_trans_time or unix_time")
  }
  train_srs20k <- parse_time(train_srs20k)
  test_srs50k  <- parse_time(test_srs50k)
  
  # Ensure we do not carry forward rows with missing parsed time
  train_srs20k <- tidyr::drop_na(train_srs20k, trans_dt)
  test_srs50k  <- tidyr::drop_na(test_srs50k,  trans_dt)
  
  # Fix wday() function parameters and set 1=Monday, 7=Sunday
  augment_time <- function(dt) {
    dt %>% mutate(
      hour = hour(trans_dt),
      # Fix wday parameter issue, set 1=Monday, 7=Sunday
      wday_num = wday(trans_dt),  # Numeric representation (1=Sunday, 7=Saturday)
      wday = case_when(
        wday_num == 1 ~ "Sun",  # Sunday
        wday_num == 2 ~ "Mon",  # Monday
        wday_num == 3 ~ "Tue",  # Tuesday
        wday_num == 4 ~ "Wed",  # Wednesday
        wday_num == 5 ~ "Thu",  # Thursday
        wday_num == 6 ~ "Fri",  # Friday
        wday_num == 7 ~ "Sat"   # Saturday
      ),
      # Adjust to 1=Monday, 7=Sunday
      wday_adjusted = case_when(
        wday_num == 2 ~ "Mon",  # Monday = 1
        wday_num == 3 ~ "Tue",  # Tuesday = 2
        wday_num == 4 ~ "Wed",  # Wednesday = 3
        wday_num == 5 ~ "Thu",  # Thursday = 4
        wday_num == 6 ~ "Fri",  # Friday = 5
        wday_num == 7 ~ "Sat",  # Saturday = 6
        wday_num == 1 ~ "Sun"   # Sunday = 7
      ),
      is_weekend = wday_adjusted %in% c("Sat","Sun"),
      month = month(trans_dt),
      daypart = dplyr::case_when(
        hour >= 0  & hour < 6  ~ "late_night",
        hour >= 6  & hour < 12 ~ "morning",
        hour >= 12 & hour < 18 ~ "afternoon",
        TRUE ~ "evening"
      )
    ) %>% 
      select(-wday_num, -wday) %>%  # Remove temporary variables and original wday
      rename(wday = wday_adjusted)   # Rename adjusted wday
  }
  train_srs20k <- augment_time(train_srs20k)
  test_srs50k  <- augment_time(test_srs50k)
  
  # 2.2 Behavioural features (card-level rolling windows)
  mk_behaviour <- function(dt) {
    if (!all(c("cc_num","trans_dt","amt") %in% names(dt)))
      stop("Missing cc_num / trans_dt / amt")
    
    # At this point we assume no NA rows (amt/trans_dt cleaned); if any slipped, we handle gracefully.
    dt <- dt %>% mutate(trans_dt_missing = is.na(trans_dt))
    dt_clean <- dt %>% filter(!is.na(trans_dt))
    if (nrow(dt_clean) == 0L) {
      return(dt %>% mutate(
        time_since_last = NA_real_,
        tx_count_1h = NA_integer_, amt_sum_1h = NA_real_,
        tx_count_24h = NA_integer_, amt_sum_24h = NA_real_
      ) %>% select(-trans_dt_missing) %>% as_tibble())
    }
    x <- as.data.table(dt_clean)[order(cc_num, trans_dt)]
    setkey(x, cc_num, trans_dt)
    x[, time_since_last := as.numeric(trans_dt - shift(trans_dt, type = "lag")), by = cc_num]
    calc_window <- function(DT, win_secs = 3600) {
      n <- nrow(DT); if (n == 0L) return(list(count=integer(0), sum_amt=numeric(0)))
      start <- integer(n); j <- 1L; ps <- c(0, cumsum(DT$amt)); tt <- DT$trans_dt
      for (i in 1L:n) {
        ti <- tt[i]
        if (!is.na(ti)) { while (j < i && !is.na(tt[j]) && tt[j] < ti - win_secs) j <- j + 1L }
        start[i] <- j
      }
      cnt <- (1L:n) - start + 1L
      ssum <- ps[(1L:n)+1L] - ps[start]
      list(count=cnt, sum_amt=ssum)
    }
    by_cc <- x[, .(cc_num, trans_dt, amt)]
    r1h  <- by_cc[, {z <- calc_window(.SD, 3600);  .(count=z$count, sum_amt=z$sum_amt)}, by=cc_num]
    r24h <- by_cc[, {z <- calc_window(.SD, 86400); .(count=z$count, sum_amt=z$sum_amt)}, by=cc_num]
    x[, `:=`(tx_count_1h=r1h$count, amt_sum_1h=r1h$sum_amt,
             tx_count_24h=r24h$count, amt_sum_24h=r24h$sum_amt)]
    feats <- as_tibble(x)[, c("cc_num","trans_dt","time_since_last","tx_count_1h","amt_sum_1h","tx_count_24h","amt_sum_24h")]
    out <- dt %>% left_join(feats, by=c("cc_num","trans_dt")) %>% select(-trans_dt_missing)
    as_tibble(out)
  }
  train_srs20k <- mk_behaviour(train_srs20k)
  test_srs50k  <- mk_behaviour(test_srs50k)
  
  # 2.3 Distance feature (customer–merchant haversine)
  merc_lat  <- if ("merchant_lat"  %in% names(train_srs20k)) "merchant_lat"  else if ("merch_lat"  %in% names(train_srs20k)) "merch_lat"  else NA
  merc_long <- if ("merchant_long" %in% names(train_srs20k)) "merchant_long" else if ("merch_long" %in% names(train_srs20k)) "merch_long" else NA
  add_distance <- function(dt) {
    # Only add the feature when all required columns exist; otherwise, do nothing to avoid introducing NA.
    if (all(c("lat","long", merc_lat, merc_long) %in% names(dt))) {
      dt %>% mutate(dist_cust_merc = as.numeric(distHaversine(
        cbind(long, lat), cbind(.data[[merc_long]], .data[[merc_lat]])
      )))
    } else dt
  }
  train_srs20k <- add_distance(train_srs20k)
  test_srs50k  <- add_distance(test_srs50k)
  
  # 2.4 High-cardinality reduction + amount transform
  top_n <- function(x, N = 100L) {
    if (!is.character(x) && !is.factor(x)) return(factor(x))
    keep <- names(sort(table(x), decreasing = TRUE))[1:min(N, length(unique(x)))]
    forcats::fct_other(factor(x), keep = keep)
  }
  if ("merchant" %in% names(train_srs20k)) {
    train_srs20k <- train_srs20k %>% mutate(merchant_top = top_n(merchant, 100))
    test_srs50k  <- test_srs50k  %>% mutate(merchant_top  = top_n(merchant, 100))
  }
  if ("job" %in% names(train_srs20k)) {
    train_srs20k <- train_srs20k %>% mutate(job_top = top_n(job, 30))
    test_srs50k  <- test_srs50k  %>% mutate(job_top  = top_n(job, 30))
  }
  train_srs20k <- train_srs20k %>% mutate(log_amt = log1p(amt))
  test_srs50k  <- test_srs50k  %>% mutate(log_amt = log1p(amt))
  
  # 2.5 Drop leakage/ID columns
  cols_drop <- c("trans_num","first","last","street","dob","cc_num",
                 "trans_date_trans_time","unix_time","city","zip","merchant")
  train_srs20k <- train_srs20k %>% select(-any_of(cols_drop))
  test_srs50k  <- test_srs50k  %>% select(-any_of(cols_drop))
  
  # ---------- Step 3) Recipe (fit on sampled TRAIN) ----------
  # Note: No NA imputation steps since we dropped NA rows upfront.
  rec <- recipe(is_fraud ~ ., data = train_srs20k) %>%
    update_role(trans_dt, new_role = "id") %>% step_rm(trans_dt) %>%
    step_mutate(wday=as.factor(wday), daypart=as.factor(daypart), is_weekend=as.factor(is_weekend)) %>%
    step_other(all_nominal_predictors(), threshold = 0.01, other = "other") %>%
    step_novel(all_nominal_predictors()) %>%
    step_unknown(all_nominal_predictors(), new_level = "Unknown") %>%  # harmless due to upfront cleaning
    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
    step_zv(all_predictors()) %>%
    step_center(all_numeric_predictors()) %>%
    step_scale(all_numeric_predictors())
  prep_rec <- prep(rec, training = train_srs20k, retain = TRUE)
  
  # ---------- Step 4) Bake ----------
  train_fe <- bake(prep_rec, new_data = train_srs20k)
  test_fe  <- bake(prep_rec, new_data = test_srs50k)
  
  # Lock outcome type to factor with levels c("no","yes") (defensive)
  train_fe$is_fraud <- factor(as.character(train_fe$is_fraud), levels = c("no","yes"))
  test_fe$is_fraud  <- factor(as.character(test_fe$is_fraud),  levels = c("no","yes"))
  
  cat(sprintf("Train FE: %d x %d | Test FE: %d x %d\n",
              nrow(train_fe), ncol(train_fe), nrow(test_fe), ncol(test_fe)))
  
  # ---------- Step 5) SMOTE (TRAIN only) ----------
  cat("\n=== APPLYING SMOTE (using ROSE package) ===\n")
  
  # Safety: ensure engineered field has no NA (first txn per card)
  if ("time_since_last" %in% names(train_fe)) {
    train_fe$time_since_last[is.na(train_fe$time_since_last)] <- 0
  }
  
  # Quick NA check before SMOTE (fail fast if any remain)
  na_any <- anyNA(train_fe)
  if (na_any) {
    bad <- sort(colSums(is.na(train_fe)), decreasing = TRUE)
    bad <- bad[bad > 0]
    print(head(bad, 10))
    stop("NAs found in train_fe. Fix NAs before SMOTE.")
  }
  
  # Display pre-SMOTE distribution
  pre_smote_counts <- table(train_fe$is_fraud)
  pre_smote_prop <- prop.table(pre_smote_counts)
  cat("Pre-SMOTE Distribution:\n")
  cat(sprintf("  - No:  %d (%.4f%%)\n", pre_smote_counts["no"], pre_smote_prop["no"] * 100))
  cat(sprintf("  - Yes: %d (%.4f%%)\n", pre_smote_counts["yes"], pre_smote_prop["yes"] * 100))
  
  # Use ROSE package for oversampling
  # Set target fraud rate (5%)
  target_fraud_rate <- 0.05
  cat(sprintf("Applying ROSE oversampling to achieve %.1f%% fraud rate...\n", target_fraud_rate * 100))
  
  # Apply ROSE oversampling
  set.seed(6104)
  train_rose <- ROSE::ovun.sample(
    is_fraud ~ ., 
    data = train_fe,
    p = target_fraud_rate,  # Target fraud rate
    seed = 6104,
    method = "over"  # Only perform oversampling
  )$data
  
  # Ensure target variable is factor
  train_rose$is_fraud <- factor(train_rose$is_fraud, levels = c("no", "yes"))
  
  # Use standard object name
  train_stratsample_smote <- train_rose
  
  # Quick consistency check
  stopifnot(is.factor(train_stratsample_smote$is_fraud),
            identical(levels(train_stratsample_smote$is_fraud), c("no","yes")))
  
  # === Detailed SMOTE Analysis ===
  cat("\n=== Detailed SMOTE Analysis ===\n")
  
  # Calculate counts before and after SMOTE
  original_counts <- table(train_fe$is_fraud)
  smote_counts <- table(train_stratsample_smote$is_fraud)
  
  # Calculate proportions
  original_prop <- prop.table(original_counts)
  smote_prop <- prop.table(smote_counts)
  
  cat("Original Data Distribution:\n")
  cat(sprintf("  - No (Non-fraud):  %d transactions (%.4f%%)\n", 
              original_counts["no"], original_prop["no"] * 100))
  cat(sprintf("  - Yes (Fraud):     %d transactions (%.4f%%)\n", 
              original_counts["yes"], original_prop["yes"] * 100))
  
  cat("\nSMOTE Data Distribution:\n")
  cat(sprintf("  - No (Non-fraud):  %d transactions (%.4f%%)\n", 
              smote_counts["no"], smote_prop["no"] * 100))
  cat(sprintf("  - Yes (Fraud):     %d transactions (%.4f%%)\n", 
              smote_counts["yes"], smote_prop["yes"] * 100))
  
  # Calculate growth factors
  growth_no <- smote_counts["no"] / original_counts["no"]
  growth_yes <- smote_counts["yes"] / original_counts["yes"]
  
  cat("\nData Growth Analysis:\n")
  cat(sprintf("  - Non-fraud growth: %.2f times\n", growth_no))
  cat(sprintf("  - Fraud growth:     %.2f times\n", growth_yes))
  
  # Calculate fraud ratio changes
  fraud_ratio_original <- original_prop["yes"]
  fraud_ratio_smote <- smote_prop["yes"]
  
  cat(sprintf("\nFraud Ratio Change: %.4f%% → %.4f%%\n", 
              fraud_ratio_original * 100, fraud_ratio_smote * 100))
  
  # Total dataset size change
  cat(sprintf("\nTotal Dataset Size Change: %d → %d transactions (%.2f times growth)\n",
              nrow(train_fe), nrow(train_stratsample_smote),
              nrow(train_stratsample_smote) / nrow(train_fe)))
  
  # Brief summary
  tab_smote <- prop.table(table(train_stratsample_smote$is_fraud))
  cat("\n=== Summary ===\n")
  cat("After SMOTE: size=", nrow(train_stratsample_smote),
      ", yes-rate=", round(tab_smote["yes"] * 100, 4), "%\n", sep = "")
  
  # ---------- Step 6) Export (CSV only) ----------
  dir.create("exports", showWarnings = FALSE)
  
  data.table::fwrite(train_stratsample_smote,  file.path("exports","train_stratsample_smote.csv"))
  data.table::fwrite(test_fe,                  file.path("exports","test_stratsample_50k_fe.csv"))
  
  # Ensure test_stratsample_50k_fe object exists
  if (!exists("test_stratsample_50k_fe")) {
    test_stratsample_50k_fe <- test_fe
  }
  
  cat("\nCSV files written to ./exports:\n")
  cat("- train_stratsample_smote.csv\n")
  cat("- test_stratsample_50k_fe.csv\n")
}

# Final confirmation
cat("\n✅ Data processing completed successfully!\n")
cat("Available objects in environment:\n")
cat("  - train_stratsample_smote:", nrow(train_stratsample_smote), "rows\n")
cat("  - test_stratsample_50k_fe:", nrow(test_stratsample_50k_fe), "rows\n")
cat("  - Final SMOTE fraud rate:", round(mean(train_stratsample_smote$is_fraud == "yes") * 100, 4), "%\n")

# ============================================
# Part 2.3 - Model 1: Decision Tree 
# Dataset object: train_stratsample_smote
# Steps: Split 80/20 -> CV -> pick cp -> refit -> validation -> summary -> human-readable plot (hi-res) -> save artifacts
# ============================================

# Load required packages
req_pkgs <- c("caret","rpart","dplyr","pROC","rpart.plot","ragg", "ggplot2")
to_install <- setdiff(req_pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)
suppressPackageStartupMessages({
  library(caret); library(rpart); library(dplyr); library(pROC)
  library(rpart.plot); library(ragg); library(ggplot2)
})
set.seed(6104)

# Input validation
# Ensure is_fraud is a factor with levels c("no","yes")
train_stratsample_smote$is_fraud <- factor(
  ifelse(tolower(as.character(train_stratsample_smote$is_fraud)) %in% c("1","yes","y","true","t"), "yes", "no"),
  levels = c("no","yes")
)

# Validate data structure
stopifnot(is.factor(train_stratsample_smote$is_fraud),
          identical(levels(train_stratsample_smote$is_fraud), c("no","yes")))
if (!exists("train_stratsample_smote")) {
  stop("Object 'train_stratsample_smote' not found. Please run data processing first.")
}

# ============================================
# NEW: Split data into 80% training and 20% validation
# ============================================
cat("=== Splitting Data into 80% Training and 20% Validation ===\n")

set.seed(6104)  # For reproducibility
train_index <- createDataPartition(train_stratsample_smote$is_fraud, 
                                   p = 0.8, 
                                   list = FALSE, 
                                   times = 1)

train_80 <- train_stratsample_smote[train_index, ]
validation_20 <- train_stratsample_smote[-train_index, ]

cat(sprintf("Training set (80%%): %d observations\n", nrow(train_80)))
cat(sprintf("Validation set (20%%): %d observations\n", nrow(validation_20)))
cat(sprintf("Class distribution in training set:\n"))
print(table(train_80$is_fraud))
cat(sprintf("Class distribution in validation set:\n"))
print(table(validation_20$is_fraud))

# Define model file paths
final_tree_file <- "models/model1_tree_final.rds"
tree_plot_file <- "models/model1_tree_structure_human.pdf"
roc_plot_file <- "models/model1_tree_roc_curve.pdf"
tuning_plot_file <- "models/model1_tree_tuning_plot.pdf"
validation_roc_plot_file <- "models/model1_tree_validation_roc.pdf"
validation_results_file <- "models/model1_tree_validation_results.pdf"

# Create models directory if it doesn't exist
dir.create("models", showWarnings = FALSE)

cat("=== Decision Tree Model Smart Loading/Training ===\n")

# ============================================
# NEW: Check if model is already uploaded in R environment
# ============================================
model_uploaded <- FALSE
if (exists("uploaded_tree_model")) {
  cat("🔍 Checking for uploaded model in R environment...\n")
  if (inherits(uploaded_tree_model, "rpart")) {
    cat("✅ Found uploaded Decision Tree model in R environment\n")
    
    # Validate uploaded model structure and variables
    model_vars <- all.vars(uploaded_tree_model$terms)
    current_vars <- names(train_80)
    missing_vars <- setdiff(model_vars, current_vars)
    
    if (length(missing_vars) == 0) {
      final_tree <- uploaded_tree_model
      model_uploaded <- TRUE
      
      # Extract cp value from uploaded model
      best_cp <- final_tree$cptable[which.min(final_tree$cptable[, "xerror"]), "CP"]
      cat("✅ Using uploaded model with cp:", best_cp, "\n")
      cat("💾 Saving uploaded model to file for future use...\n")
      saveRDS(final_tree, file = final_tree_file)
    } else {
      cat("⚠️ Uploaded model has incompatible variables:", paste(missing_vars, collapse = ", "), "\n")
      cat("🔄 Ignoring uploaded model due to feature mismatch...\n")
    }
  } else {
    cat("⚠️ Uploaded object is not a valid rpart model\n")
  }
}

# 1) Try to load pre-trained model; train if not exists and no uploaded model
model_loaded <- FALSE

if (!model_uploaded && file.exists(final_tree_file)) {
  cat("📁 Loading pre-trained Decision Tree model...\n")
  
  tryCatch({
    final_tree <- readRDS(final_tree_file)
    
    # Validate loaded model structure
    if (inherits(final_tree, "rpart")) {
      cat("✅ Decision Tree model successfully loaded\n")
      
      # === CHECK FOR MISSING VARIABLES ===
      model_vars <- all.vars(final_tree$terms)
      current_vars <- names(train_80)
      missing_vars <- setdiff(model_vars, current_vars)
      
      if (length(missing_vars) > 0) {
        cat("⚠️ Missing variables in current data:", paste(missing_vars, collapse = ", "), "\n")
        cat("🔄 Forcing model retraining due to feature mismatch...\n")
        model_loaded <- FALSE
        
        # Remove the problematic model file to prevent future issues
        if (file.exists(final_tree_file)) {
          file.remove(final_tree_file)
          cat("🗑️ Removed incompatible model file:", final_tree_file, "\n")
        }
      } else {
        model_loaded <- TRUE
        
        # Extract cp value from loaded model
        best_cp <- final_tree$cptable[which.min(final_tree$cptable[, "xerror"]), "CP"]
        cat("Best cp from loaded model:", best_cp, "\n")
      }
    } else {
      cat("⚠️ Warning: Loaded model has incorrect structure. Retraining...\n")
      model_loaded <- FALSE
    }
  }, error = function(e) {
    cat("❌ Error loading model:", e$message, "\n")
    cat("🔄 Retraining model...\n")
    model_loaded <- FALSE
  })
} else if (!model_uploaded) {
  cat("📭 No saved Decision Tree model found\n")
  model_loaded <- FALSE
}

# 2) Train model if not uploaded and not loaded successfully
if (!model_uploaded && !model_loaded) {
  cat("🔧 Starting Decision Tree model training...\n")
  
  # 10-fold cross-validation to tune cp parameter (using train_80)
  ctrl <- trainControl(method = "cv", number = 10,
                       classProbs = TRUE, summaryFunction = twoClassSummary)
  cp_grid <- expand.grid(cp = c(0.001, 0.003, 0.005, 0.01, 0.02))
  
  set.seed(6104)
  tree_cv <- caret::train(
    is_fraud ~ .,
    data = train_80,
    method = "rpart",
    trControl = ctrl,
    metric = "ROC",
    tuneGrid = cp_grid
  )
  
  # Extract best cp value from cross-validation
  best_cp <- tree_cv$bestTune$cp
  cat("Best cp from 10-fold CV:", best_cp, "\n")
  
  # Refit final model with best cp (using train_80)
  final_tree <- rpart::rpart(
    is_fraud ~ .,
    data = train_80,
    control = rpart.control(cp = best_cp)
  )
  
  # Save trained model
  saveRDS(final_tree, file = final_tree_file)
  cat("💾 Decision Tree model saved:", final_tree_file, "\n")
} else if (model_uploaded) {
  cat("📥 Using uploaded model, skipping training phase\n")
} else if (model_loaded) {
  cat("📁 Using pre-loaded model, skipping training phase\n")
}

# 3) Generate model summary (text only)
get_depth <- function(obj) {
  fr <- obj$frame
  max(sapply(path.rpart(obj, nodes = as.numeric(row.names(fr))), length)) - 1
}
depth_val <- get_depth(final_tree)
n_nodes <- nrow(final_tree$frame)

# Calculate variable importance from final_tree
vi_final <- final_tree$variable.importance
if (!is.null(vi_final)) {
  vi_df <- data.frame(
    feature = names(vi_final),
    Importance = as.numeric(vi_final)
  ) %>% arrange(desc(Importance))
} else {
  vi_df <- NULL
}

# Print model summary
cat("\n=== Model Summary: Decision Tree (rpart) ===\n")
cat(sprintf("- Selected cp: %.4f\n", best_cp))
cat(sprintf("- Tree depth: %d\n", depth_val))
cat(sprintf("- Number of nodes: %d\n", n_nodes))
if (!is.null(vi_df) && nrow(vi_df) > 0) {
  vi_top <- vi_df %>% head(10)
  cat("- Top features (by importance):\n")
  for (i in seq_len(nrow(vi_top))) {
    cat(sprintf("  * %s: %.3f\n", vi_top$feature[i], as.numeric(vi_top$Importance[i])))
  }
} else {
  cat("- Variable importance not available.\n")
}

# 4) Generate human-readable tree plot (only if not exists or model was trained/uploaded)
if ((!file.exists(tree_plot_file)) || model_uploaded || (!model_loaded && !model_uploaded)) {
  cat("\n🌳 Generating decision tree visualization...\n")
  
  # 4.1 Calculate mean and standard deviation from original training data (train_80) to restore original values
  cat("Calculating variable means and standard deviations to restore original values...\n")
  
  # Calculate mean and standard deviation for numeric variables from training data (train_80)
  numeric_vars <- c("amt", "amt_sum_1h", "amt_sum_24h", "hour", "time_since_last")
  numeric_vars <- numeric_vars[numeric_vars %in% names(train_80)]
  
  means <- sapply(train_80[numeric_vars], function(x) mean(x, na.rm = TRUE))
  sds <- sapply(train_80[numeric_vars], function(x) sd(x, na.rm = TRUE))
  
  cat("Calculated means and standard deviations:\n")
  for (var in names(means)) {
    cat(sprintf("  %s: mean = %.2f, standard deviation = %.2f\n", var, means[var], sds[var]))
  }
  
  # 4.2 Define human-readable variable names for display
  pretty_names <- c(
    amt = "Transaction amount",
    log_amt = "Transaction amount(log)",
    amt_sum_1h = "Total amount in 1 hour", 
    amt_sum_24h = "Total amount in 24 hours",
    hour = "Transaction time",
    time_since_last = "Gap since last (s)",
    is_weekend = "Is weekend"
  )
  
  # 4.3 Define formatting helper functions
  fmt_money <- function(x) {
    if (is.na(x) || !is.finite(x)) return("NA")
    d <- ifelse(abs(x) >= 1000, 0, 2)
    paste0("US$", formatC(x, format = "f", big.mark = ",", digits = d))
  }
  
  fmt_num <- function(x) {
    if (is.na(x) || !is.finite(x)) return("NA")
    d <- ifelse(abs(x) >= 10, 0, 2)
    formatC(x, format = "f", digits = d)
  }
  
  # Duration formatting function (seconds → seconds/minutes/hours)
  fmt_duration <- function(x) {
    x <- max(x, 0)
    if (x < 60) paste0(round(x), " seconds") 
    else if (x < 3600) paste0(round(x / 60), " minutes") 
    else paste0(round(x / 3600, 1), " hours")
  }
  
  # Hour formatting function (0-23)
  fmt_hour <- function(x) {
    h <- max(min(round(x), 23), 0)
    paste0(h, ":00")
  }
  
  # 4.4 Custom split label function to convert z-scores to human-readable values
  split_fun_human <- function(x, labs, digits, varlen, faclen) {
    out <- labs
    for (i in seq_along(labs)) {
      lab <- labs[i]
      # Parse split condition: variable name, operator, threshold
      m <- regexec("^\\s*([A-Za-z0-9_\\.]+)\\s*([<>=]+)\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*$", lab)
      parts <- regmatches(lab, m)[[1]]
      
      if (length(parts) == 4) {
        var <- parts[2]
        op <- parts[3]
        thr_z <- as.numeric(parts[4])
        
        # Get human-readable variable name
        var_nice <- ifelse(var %in% names(pretty_names), pretty_names[[var]], var)
        
        # Restore original value
        if (!is.null(means) && !is.null(sds) &&
            var %in% names(means) && var %in% names(sds) &&
            !is.na(sds[[var]]) && sds[[var]] > 0) {
          
          thr_original <- means[[var]] + thr_z * sds[[var]]
          
          # Apply appropriate formatting based on variable type
          if (grepl("^amt", var)) {
            thr_txt <- fmt_money(max(thr_original, 0))
          } else if (var %in% c("time_since_last")) {
            thr_txt <- fmt_duration(thr_original)
          } else if (var %in% c("hour")) {
            thr_txt <- fmt_hour(thr_original)
          } else {
            thr_txt <- fmt_num(thr_original)
          }
          
          out[i] <- paste0(var_nice, " ", op, " ", thr_txt)
          
        } else {
          ztxt <- if (thr_z >= 0) {
            paste0("mean+", fmt_num(thr_z), "SD")
          } else {
            paste0("mean-", fmt_num(abs(thr_z)), "SD")
          }
          out[i] <- paste0(var_nice, " ", op, " (", ztxt, ")")
        }
      } else {
        out[i] <- gsub("\\s+", " ", lab)
      }
    }
    out
  }
  
  # 4.5 Optional: lightly prune tree for better readability in plot
  prune_cp  <- best_cp * 1.5
  plot_tree <- tryCatch(rpart::prune(final_tree, cp = prune_cp), error = function(e) final_tree)
  
  # 4.6 Configure plot settings with custom branch labels
  plot_args <- list(
    type = 2,
    extra = 0,
    fallen.leaves = TRUE,
    under = FALSE,
    faclen = 0,
    cex = 1,
    split.cex = 1,
    tweak = 1,
    branch.lwd = 1.5,
    branch.col = "black",
    shadow.col = NA,
    nn = FALSE,
    compress = TRUE,
    uniform = TRUE,
    yesno = TRUE,
    split.fun = split_fun_human,
    clip.right.labs = FALSE,
    leaf.round = 0,
    box.col = "pink",
    border.col = "lightgray",
    gap = 1,
    minbranch = 1,
    split.col = "black",
    split.box.col = "white",
    split.border.col = "black",
    split.round = 0,
    split.shadow.col = NA
  )
  
  # Generate PDF output
  pdf(file = tree_plot_file, width = 30, height = 20, family = "Helvetica")
  do.call(rpart.plot::rpart.plot, c(list(x = plot_tree), plot_args))
  
  # Add explanatory text
  text(x = 0.5, y = -0.05,
       labels = "Node content: Predicted class (fraud probability)\nBranches: Decision paths based on split conditions",
       cex = 0.8, xpd = NA)
  
  dev.off()
  
  cat("✅ Generated decision tree chart with original values:", tree_plot_file, "\n")
} else {
  cat("📊 Decision tree plot already exists:", tree_plot_file, "\n")
}

# 5) Generate Parameter Tuning Plot (only if we have the CV object from training)
if (!model_uploaded && !model_loaded && exists("tree_cv") && !file.exists(tuning_plot_file)) {
  cat("\n📊 Generating parameter tuning plot...\n")
  
  # Extract cross-validation results
  tuning_results <- tree_cv$results
  
  # Create tuning plot
  tuning_plot <- ggplot(tuning_results, aes(x = cp, y = ROC)) +
    geom_line(color = "steelblue", size = 1.2) +
    geom_point(color = "red", size = 3) +
    geom_vline(xintercept = best_cp, linetype = "dashed", color = "darkred", alpha = 0.7) +
    geom_point(data = tuning_results[tuning_results$cp == best_cp, ], 
               color = "darkred", size = 4, shape = 21, fill = "white", stroke = 2) +
    labs(
      title = "Decision Tree Parameter Tuning - Complexity Parameter vs AUC",
      subtitle = sprintf("Best cp = %.4f (AUC = %.4f)", best_cp, max(tuning_results$ROC)),
      x = "Complexity Parameter (cp)",
      y = "AUC (Cross-Validation)"
    ) +
    scale_x_continuous(breaks = tuning_results$cp) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 12),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_line(color = "grey95"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  # Save tuning plot
  ggsave(tuning_plot_file, tuning_plot, width = 10, height = 6)
  cat("✅ Parameter tuning plot saved:", tuning_plot_file, "\n")
} else if (file.exists(tuning_plot_file)) {
  cat("📊 Parameter tuning plot already exists:", tuning_plot_file, "\n")
} else if (model_uploaded || model_loaded) {
  cat("📊 Note: Parameter tuning plot not available for uploaded/loaded model (requires training data)\n")
}

# 6) Generate ROC Curve for Decision Tree on Training Set
cat("\n=== Generating ROC Curve for Decision Tree (Training Set) ===\n")

# Use the final_tree model for prediction on training set
tree_probs_train <- predict(final_tree, newdata = train_80, type = "prob")

# Calculate ROC curve for training set
tree_roc_train <- roc(response = train_80$is_fraud, 
                      predictor = tree_probs_train[, "yes"])

# Calculate AUC value for training set
tree_auc_train <- auc(tree_roc_train)
cat(sprintf("Decision Tree Training AUC: %.4f\n", tree_auc_train))

# Create ROC curve plot for training set
roc_plot_train <- ggroc(tree_roc_train, alpha = 0.8, color = "red", linewidth = 1.2) +
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", alpha = 0.5) +
  labs(title = "Decision Tree - ROC Curve (Training Set)",
       subtitle = sprintf("AUC = %.4f", tree_auc_train),
       x = "Specificity",
       y = "Sensitivity") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_line(color = "grey95")
  ) +
  coord_equal()

# Save ROC plot for training set
ggsave(roc_plot_file, roc_plot_train, width = 8, height = 6)
cat("📊 Training ROC curve saved:", roc_plot_file, "\n")

# ============================================
# Validation on 20% holdout set
# ============================================
cat("\n=== Validating Decision Tree on 20% Holdout Set ===\n")

# Use the final_tree model for prediction on validation set
tree_probs_validation <- predict(final_tree, newdata = validation_20, type = "prob")

# Calculate ROC curve for validation set
tree_roc_validation <- roc(response = validation_20$is_fraud, 
                           predictor = tree_probs_validation[, "yes"])

# Calculate AUC value for validation set
tree_auc_validation <- auc(tree_roc_validation)
cat(sprintf("Decision Tree Validation AUC: %.4f\n", tree_auc_validation))

# Create ROC curve plot for validation set
roc_plot_validation <- ggroc(tree_roc_validation, alpha = 0.8, color = "blue", linewidth = 1.2) +
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", alpha = 0.5) +
  labs(title = "Decision Tree - ROC Curve (Validation Set)",
       subtitle = sprintf("AUC = %.4f", tree_auc_validation),
       x = "Specificity",
       y = "Sensitivity") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_line(color = "grey95")
  ) +
  coord_equal()

# Save ROC plot for validation set
ggsave(validation_roc_plot_file, roc_plot_validation, width = 8, height = 6)
cat("📊 Validation ROC curve saved:", validation_roc_plot_file, "\n")

# ============================================
# Create comprehensive validation results PDF
# ============================================
cat("\n=== Generating Comprehensive Validation Results PDF ===\n")

pdf(file = validation_results_file, width = 11, height = 8.5)

# Page 1: Validation Performance Summary
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)
plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), 
     xlab = "", ylab = "", axes = FALSE, main = "Decision Tree Validation Results")
text(0.5, 0.9, "Decision Tree Model Validation Summary", cex = 1.5, font = 2)

# Model source information
model_source <- if (model_uploaded) {
  "Uploaded Model"
} else if (model_loaded) {
  "Pre-loaded Model"
} else {
  "Newly Trained Model"
}
text(0.1, 0.85, sprintf("Model Source: %s", model_source), pos = 4, cex = 1.2)

# Performance metrics
text(0.1, 0.7, sprintf("Training Set Size: %d observations", nrow(train_80)), pos = 4, cex = 1.2)
text(0.1, 0.65, sprintf("Validation Set Size: %d observations", nrow(validation_20)), pos = 4, cex = 1.2)
text(0.1, 0.6, sprintf("Training AUC: %.4f", tree_auc_train), pos = 4, cex = 1.2)
text(0.1, 0.55, sprintf("Validation AUC: %.4f", tree_auc_validation), pos = 4, cex = 1.2)
text(0.1, 0.5, sprintf("Best cp parameter: %.4f", best_cp), pos = 4, cex = 1.2)
text(0.1, 0.45, sprintf("Tree Depth: %d", depth_val), pos = 4, cex = 1.2)
text(0.1, 0.4, sprintf("Number of Nodes: %d", n_nodes), pos = 4, cex = 1.2)

# Performance comparison
performance_diff <- tree_auc_train - tree_auc_validation
if (performance_diff > 0.05) {
  performance_note <- "⚠️ Significant overfitting detected"
} else if (performance_diff > 0.02) {
  performance_note <- "ℹ️ Moderate overfitting detected"
} else {
  performance_note <- "✅ Good generalization performance"
}

text(0.1, 0.3, sprintf("Performance Difference (Train - Val): %.4f", performance_diff), pos = 4, cex = 1.2)
text(0.1, 0.25, performance_note, pos = 4, cex = 1.2, col = ifelse(performance_diff > 0.05, "red", "darkgreen"))

# Page 2: Side-by-side ROC curves
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
# Training ROC
plot(tree_roc_train, main = "Training ROC Curve", col = "red", lwd = 2)
text(0.5, 0.3, sprintf("AUC = %.4f", tree_auc_train), cex = 1.2)
# Validation ROC
plot(tree_roc_validation, main = "Validation ROC Curve", col = "blue", lwd = 2)
text(0.5, 0.3, sprintf("AUC = %.4f", tree_auc_validation), cex = 1.2)

# Page 3: Variable Importance (if available)
if (!is.null(vi_df) && nrow(vi_df) > 0) {
  vi_top_10 <- head(vi_df, 10)
  par(mfrow = c(1, 1), mar = c(10, 5, 4, 2) + 0.1)  # bottom margin
  barplot(
    vi_top_10$Importance,
    names.arg = vi_top_10$feature,
    las = 2,  
    col = "steelblue",
    main = "Top 10 Feature Importance",
    ylab = "Importance Score",
    cex.names = 1  #  label 
  )
}

dev.off()
cat("✅ Comprehensive validation results saved:", validation_results_file, "\n")

# Display AUC confidence intervals for both sets
ci_auc_train <- ci.auc(tree_roc_train)
ci_auc_validation <- ci.auc(tree_roc_validation)

cat("\n=== Decision Tree ROC Analysis ===\n")
cat(sprintf("- Training AUC: %.4f (95%% CI: [%.4f, %.4f])\n", tree_auc_train, ci_auc_train[1], ci_auc_train[3]))
cat(sprintf("- Validation AUC: %.4f (95%% CI: [%.4f, %.4f])\n", tree_auc_validation, ci_auc_validation[1], ci_auc_validation[3]))
cat(sprintf("- Performance Difference: %.4f\n", performance_diff))
cat("- ROC curve visualizations:\n")
cat("  -", roc_plot_file, "(training)\n")
cat("  -", validation_roc_plot_file, "(validation)\n")
cat("  -", validation_results_file, "(comprehensive results)\n")

# 7) Final artifacts summary
cat("\n=== Decision Tree Artifacts ===\n")
if (model_uploaded) {
  cat("📥 Model uploaded from R environment:\n")
} else if (model_loaded) {
  cat("📁 Model loaded from storage:\n")
} else {
  cat("🔧 Model trained and saved:\n")
}
cat("   -", final_tree_file, "(final rpart model)\n")
cat("   -", tree_plot_file, "(decision tree, human-readable)\n")
if (file.exists(tuning_plot_file)) {
  cat("   -", tuning_plot_file, "(parameter tuning plot)\n")
}
cat("   -", roc_plot_file, "(training ROC curve)\n")
cat("   -", validation_roc_plot_file, "(validation ROC curve)\n")
cat("   -", validation_results_file, "(comprehensive validation results)\n")

# 8) Display model details
cat("\n=== Decision Tree Rules ===\n")
print(final_tree)

cat("\n=== Detailed Tree Summary ===\n")
summary(final_tree)

# Final success message
cat("\n✅ Decision Tree processing completed successfully!\n")
if (model_uploaded) {
  cat("📥 Model was uploaded from R environment\n")
} else if (model_loaded) {
  cat("📁 Model was loaded from existing file\n")
} else {
  cat("🔧 Model was trained and saved\n")
}
cat("📊 Best cp parameter:", best_cp, "\n")
cat("🎯 Training AUC:", round(tree_auc_train, 4), "\n")
cat("🎯 Validation AUC:", round(tree_auc_validation, 4), "\n")
cat("📈 Performance Difference:", round(performance_diff, 4), "\n")
cat("🌳 Tree depth:", depth_val, "\n")
cat("📊 Number of nodes:", n_nodes, "\n")
cat("📁 Data split: 80% training, 20% validation\n")

# ============================================
# Part 2.4 - Model 2: k-Nearest Neighbors (kNN)
# Steps: Split 80/20 -> CV -> validation -> summary -> save artifacts
# ============================================

# Load required packages
req_pkgs <- c("caret", "ggplot2", "pROC", "dplyr")
to_install <- setdiff(req_pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

suppressPackageStartupMessages({
  library(caret)
  library(ggplot2)
  library(pROC)
  library(dplyr)
})

set.seed(6104)  # Consistent with previous steps

# Input validation
stopifnot(exists("train_stratsample_smote"))

# Ensure target variable format is correct
train_stratsample_smote$is_fraud <- factor(
  ifelse(tolower(as.character(train_stratsample_smote$is_fraud)) %in% c("1","yes","y","true","t"), "yes", "no"),
  levels = c("no", "yes")
)

# ============================================
# NEW: Split data into 80% training and 20% validation
# ============================================
cat("=== Splitting Data into 80% Training and 20% Validation ===\n")

set.seed(6104)  # For reproducibility
train_index <- createDataPartition(train_stratsample_smote$is_fraud, 
                                   p = 0.8, 
                                   list = FALSE, 
                                   times = 1)

train_80 <- train_stratsample_smote[train_index, ]
validation_20 <- train_stratsample_smote[-train_index, ]

cat(sprintf("Training set (80%%): %d observations\n", nrow(train_80)))
cat(sprintf("Validation set (20%%): %d observations\n", nrow(validation_20)))
cat(sprintf("Class distribution in training set:\n"))
print(table(train_80$is_fraud))
cat(sprintf("Class distribution in validation set:\n"))
print(table(validation_20$is_fraud))

# Define file paths
model_path <- "models/model2_knn_final.rds"
tuning_plot_path <- "models/model2_knn_tuning_plot.pdf"
roc_plot_path <- "models/model2_knn_roc_curve.pdf"
validation_roc_plot_path <- "models/model2_knn_validation_roc.pdf"  # NEW: Validation ROC
validation_results_path <- "models/model2_knn_validation_results.pdf"  # NEW: Validation results

# Create models directory if it doesn't exist
dir.create("models", showWarnings = FALSE)

# Check ctrl
if (!exists("ctrl")) {
  cat("Note: ctrl not found, redefining trainControl for kNN...\n")
  ctrl <- trainControl(
    method = "cv", 
    number = 10,
    classProbs = TRUE, 
    summaryFunction = twoClassSummary,
    savePredictions = "final"
  )
} else {
  cat("Using existing ctrl object for kNN...\n")
}

cat("=== kNN Model Smart Loading/Training ===\n")

# ============================================
# NEW: Check if model is already uploaded in R environment
# ============================================
model_uploaded <- FALSE
if (exists("uploaded_knn_model")) {
  cat("🔍 Checking for uploaded model in R environment...\n")
  if (inherits(uploaded_knn_model, "train")) {
    cat("✅ Found uploaded kNN model in R environment\n")
    
    # Validate uploaded model structure
    model_vars <- all.vars(uploaded_knn_model$terms)
    current_vars <- names(train_80)
    missing_vars <- setdiff(model_vars, current_vars)
    
    if (length(missing_vars) == 0) {
      fit_knn <- uploaded_knn_model
      model_uploaded <- TRUE
      
      # Extract best parameters from uploaded model
      best_k <- fit_knn$bestTune$k
      cat("✅ Using uploaded model with k:", best_k, "\n")
      cat("💾 Saving uploaded model to file for future use...\n")
      saveRDS(fit_knn, file = model_path)
    } else {
      cat("⚠️ Uploaded model has incompatible variables:", paste(missing_vars, collapse = ", "), "\n")
      cat("🔄 Ignoring uploaded model due to feature mismatch...\n")
    }
  } else {
    cat("⚠️ Uploaded object is not a valid caret train model\n")
  }
}

# Function to train kNN model
train_knn <- function() {
  cat("Training kNN model...\n")
  fit_knn <- train(
    is_fraud ~ .,
    data = train_80,  # Use 80% training data
    method = "knn",
    preProcess = "scale",  # kNN requires standardization!
    trControl = ctrl,      # Use same CV settings as Decision Tree
    tuneLength = 10,       # Let caret automatically test 10 different k values
    metric = "ROC"         # Use AUC as evaluation metric
  )
  return(fit_knn)
}

# 1) Try to load pre-trained model; train if not exists and no uploaded model
model_loaded <- FALSE

if (!model_uploaded && file.exists(model_path)) {
  cat("📁 Loading pre-trained kNN model...\n")
  
  tryCatch({
    fit_knn <- readRDS(model_path)
    
    # Validate loaded model structure
    if (inherits(fit_knn, "train")) {
      cat("✅ kNN model successfully loaded\n")
      
      # === CHECK FOR MISSING VARIABLES ===
      model_vars <- all.vars(fit_knn$terms)
      current_vars <- names(train_80)
      missing_vars <- setdiff(model_vars, current_vars)
      
      if (length(missing_vars) > 0) {
        cat("⚠️ Missing variables in current data:", paste(missing_vars, collapse = ", "), "\n")
        cat("🔄 Forcing model retraining due to feature mismatch...\n")
        model_loaded <- FALSE
        
        # Remove the problematic model file to prevent future issues
        if (file.exists(model_path)) {
          file.remove(model_path)
          cat("🗑️ Removed incompatible model file:", model_path, "\n")
        }
      } else {
        model_loaded <- TRUE
        
        # Extract best parameters from loaded model
        best_k <- fit_knn$bestTune$k
        cat("Best k from loaded model:", best_k, "\n")
      }
    } else {
      cat("⚠️ Warning: Loaded model has incorrect structure. Retraining...\n")
      model_loaded <- FALSE
    }
  }, error = function(e) {
    cat("❌ Error loading model:", e$message, "\n")
    cat("🔄 Retraining model...\n")
    model_loaded <- FALSE
  })
} else if (!model_uploaded) {
  cat("📭 No saved kNN model found\n")
  model_loaded <- FALSE
}

# 2) Train model if not uploaded and not loaded successfully
if (!model_uploaded && !model_loaded) {
  cat("🔧 Starting kNN model training...\n")
  fit_knn <- train_knn()
  
  # Save trained model
  saveRDS(fit_knn, file = model_path)
  cat("💾 kNN model saved:", model_path, "\n")
} else if (model_uploaded) {
  cat("📥 Using uploaded model, skipping training phase\n")
} else if (model_loaded) {
  cat("📁 Using pre-loaded model, skipping training phase\n")
}

# Function to generate tuning plot
generate_tuning_plot <- function(fit_knn) {
  if (!exists("fit_knn") || is.null(fit_knn$results)) {
    cat("⚠️ Cannot generate tuning plot: model results not available\n")
    return(NULL)
  }
  
  knn_plot <- ggplot(fit_knn) +
    ggtitle("kNN Parameter Tuning - k Value vs AUC") +
    xlab("k Value (Number of Neighbors)") +
    ylab("AUC (Cross-Validation)") +
    theme_minimal()
  
  ggsave(tuning_plot_path, knn_plot, width = 8, height = 6)
  cat("📊 kNN parameter tuning plot saved:", tuning_plot_path, "\n")
  
  return(knn_plot)
}

# Function to generate ROC curves for both training and validation
generate_roc_curves <- function(fit_knn) {
  cat("\n=== Generating ROC Curves for kNN Model ===\n")
  
  # Training set ROC
  knn_probs_train <- predict(fit_knn, newdata = train_80, type = "prob")
  knn_roc_train <- roc(response = train_80$is_fraud, 
                       predictor = knn_probs_train[, "yes"])
  knn_auc_train <- auc(knn_roc_train)
  cat(sprintf("kNN Training AUC: %.4f\n", knn_auc_train))
  
  # Create training ROC curve plot
  knn_roc_plot_train <- ggroc(knn_roc_train, alpha = 0.8, color = "blue", linewidth = 1.2) +
    geom_abline(intercept = 1, slope = 1, linetype = "dashed", alpha = 0.5) +
    labs(title = "k-Nearest Neighbors - ROC Curve (Training Set)",
         subtitle = sprintf("AUC = %.4f", knn_auc_train),
         x = "Specificity",
         y = "Sensitivity") +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 12),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_line(color = "grey95")
    ) +
    coord_equal()
  
  # Save training ROC plot
  ggsave(roc_plot_path, knn_roc_plot_train, width = 8, height = 6)
  cat("📊 Training ROC curve saved:", roc_plot_path, "\n")
  
  # ============================================
  # NEW: Validation set ROC
  # ============================================
  knn_probs_validation <- predict(fit_knn, newdata = validation_20, type = "prob")
  knn_roc_validation <- roc(response = validation_20$is_fraud, 
                            predictor = knn_probs_validation[, "yes"])
  knn_auc_validation <- auc(knn_roc_validation)
  cat(sprintf("kNN Validation AUC: %.4f\n", knn_auc_validation))
  
  # Create validation ROC curve plot
  knn_roc_plot_validation <- ggroc(knn_roc_validation, alpha = 0.8, color = "purple", linewidth = 1.2) +
    geom_abline(intercept = 1, slope = 1, linetype = "dashed", alpha = 0.5) +
    labs(title = "k-Nearest Neighbors - ROC Curve (Validation Set)",
         subtitle = sprintf("AUC = %.4f", knn_auc_validation),
         x = "Specificity",
         y = "Sensitivity") +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 12),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_line(color = "grey95")
    ) +
    coord_equal()
  
  # Save validation ROC plot
  ggsave(validation_roc_plot_path, knn_roc_plot_validation, width = 8, height = 6)
  cat("📊 Validation ROC curve saved:", validation_roc_plot_path, "\n")
  
  # Display AUC confidence intervals
  knn_ci_auc_train <- ci.auc(knn_roc_train)
  knn_ci_auc_validation <- ci.auc(knn_roc_validation)
  
  cat(sprintf("Training AUC 95%% Confidence Interval: [%.4f, %.4f]\n", 
              knn_ci_auc_train[1], knn_ci_auc_train[3]))
  cat(sprintf("Validation AUC 95%% Confidence Interval: [%.4f, %.4f]\n", 
              knn_ci_auc_validation[1], knn_ci_auc_validation[3]))
  
  return(list(
    roc_plot_train = knn_roc_plot_train, 
    auc_train = knn_auc_train, 
    ci_auc_train = knn_ci_auc_train,
    roc_plot_validation = knn_roc_plot_validation,
    auc_validation = knn_auc_validation,
    ci_auc_validation = knn_ci_auc_validation,
    roc_train = knn_roc_train,           # 返回 ROC 对象
    roc_validation = knn_roc_validation   # 返回 ROC 对象
  ))
}

# ============================================
# NEW: Generate comprehensive validation results PDF
# ============================================
generate_validation_results <- function(fit_knn, roc_results) {
  cat("\n=== Generating Comprehensive Validation Results PDF ===\n")
  
  pdf(file = validation_results_path, width = 11, height = 8.5)
  
  # Page 1: Validation Performance Summary
  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)
  plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), 
       xlab = "", ylab = "", axes = FALSE, main = "kNN Validation Results")
  text(0.5, 0.9, "k-Nearest Neighbors Model Validation Summary", cex = 1.5, font = 2)
  
  # Model source information
  model_source <- if (model_uploaded) {
    "Uploaded Model"
  } else if (model_loaded) {
    "Pre-loaded Model"
  } else {
    "Newly Trained Model"
  }
  text(0.1, 0.85, sprintf("Model Source: %s", model_source), pos = 4, cex = 1.2)
  
  # Performance metrics
  text(0.1, 0.7, sprintf("Training Set Size: %d observations", nrow(train_80)), pos = 4, cex = 1.2)
  text(0.1, 0.65, sprintf("Validation Set Size: %d observations", nrow(validation_20)), pos = 4, cex = 1.2)
  text(0.1, 0.6, sprintf("Training AUC: %.4f", roc_results$auc_train), pos = 4, cex = 1.2)
  text(0.1, 0.55, sprintf("Validation AUC: %.4f", roc_results$auc_validation), pos = 4, cex = 1.2)
  text(0.1, 0.5, sprintf("Best k parameter: %d", fit_knn$bestTune$k), pos = 4, cex = 1.2)
  text(0.1, 0.45, sprintf("Number of Features: %d", ncol(train_80) - 1), pos = 4, cex = 1.2)
  
  # Performance comparison
  performance_diff <- roc_results$auc_train - roc_results$auc_validation
  if (performance_diff > 0.05) {
    performance_note <- "⚠️ Significant overfitting detected"
  } else if (performance_diff > 0.02) {
    performance_note <- "ℹ️ Moderate overfitting detected"
  } else {
    performance_note <- "✅ Good generalization performance"
  }
  
  text(0.1, 0.35, sprintf("Performance Difference (Train - Val): %.4f", performance_diff), pos = 4, cex = 1.2)
  text(0.1, 0.3, performance_note, pos = 4, cex = 1.2, col = ifelse(performance_diff > 0.05, "red", "darkgreen"))
  
  # Confidence intervals
  text(0.1, 0.2, sprintf("Training AUC 95%% CI: [%.4f, %.4f]", 
                         roc_results$ci_auc_train[1], roc_results$ci_auc_train[3]), pos = 4, cex = 1.1)
  text(0.1, 0.15, sprintf("Validation AUC 95%% CI: [%.4f, %.4f]", 
                          roc_results$ci_auc_validation[1], roc_results$ci_auc_validation[3]), pos = 4, cex = 1.1)
  
  # Page 2: Side-by-side ROC curves
  par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
  # Training ROC - 使用从 roc_results 传递过来的 ROC 对象
  plot(roc_results$roc_train, main = "Training ROC Curve", col = "blue", lwd = 2)
  text(0.5, 0.3, sprintf("AUC = %.4f", roc_results$auc_train), cex = 1.2)
  # Validation ROC - 使用从 roc_results 传递过来的 ROC 对象
  plot(roc_results$roc_validation, main = "Validation ROC Curve", col = "purple", lwd = 2)
  text(0.5, 0.3, sprintf("AUC = %.4f", roc_results$auc_validation), cex = 1.2)
  
  # Page 3: Parameter tuning results (if available)
  if (!is.null(fit_knn$results) && nrow(fit_knn$results) > 0) {
    par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)
    
    # Create a simple line plot showing AUC by k value
    tuning_data <- fit_knn$results
    if ("k" %in% names(tuning_data) && "ROC" %in% names(tuning_data)) {
      plot(tuning_data$k, tuning_data$ROC, type = "b", 
           col = "steelblue", lwd = 2, pch = 19,
           main = "kNN Parameter Tuning Results",
           xlab = "k Value (Number of Neighbors)", 
           ylab = "AUC")
      abline(v = fit_knn$bestTune$k, col = "red", lty = 2, lwd = 2)
      legend("topright", legend = paste("Best k =", fit_knn$bestTune$k), 
             col = "red", lty = 2, lwd = 2)
    }
  }
  
  dev.off()
  cat("✅ Comprehensive validation results saved:", validation_results_path, "\n")
  
  return(performance_diff)
}

# Function to display model summary
display_model_summary <- function(fit_knn, roc_results, performance_diff) {
  cat("\n=== kNN Model Summary ===\n")
  cat("Best k value:", fit_knn$bestTune$k, "\n")
  
  if (!is.null(fit_knn$results)) {
    cv_auc <- max(fit_knn$results$ROC, na.rm = TRUE)
    cat("Cross-validation AUC:", round(cv_auc, 4), "\n")
    
    # Get the row with best k value
    best_row <- fit_knn$results[fit_knn$results$k == fit_knn$bestTune$k, ]
    if (nrow(best_row) > 0) {
      cat("Sensitivity:", round(best_row$Sens, 4), "\n")
      cat("Specificity:", round(best_row$Spec, 4), "\n")
    }
  }
  
  cat("Number of features used:", ncol(train_80) - 1, "\n") # minus target variable
  cat("Training AUC:", round(roc_results$auc_train, 4), "\n")
  cat("Validation AUC:", round(roc_results$auc_validation, 4), "\n")
  cat("Performance Difference:", round(performance_diff, 4), "\n")
  cat(sprintf("Training AUC 95%% CI: [%.4f, %.4f]\n", 
              roc_results$ci_auc_train[1], roc_results$ci_auc_train[3]))
  cat(sprintf("Validation AUC 95%% CI: [%.4f, %.4f]\n", 
              roc_results$ci_auc_validation[1], roc_results$ci_auc_validation[3]))
}

# Main execution logic
# Generate tuning plot (only if we trained the model or have CV results)
if ((!model_uploaded && !model_loaded) || (exists("fit_knn") && !is.null(fit_knn$results))) {
  generate_tuning_plot(fit_knn)
} else {
  cat("📊 Parameter tuning plot not available for uploaded/loaded model\n")
}

# Generate ROC curves and validation results
roc_results <- generate_roc_curves(fit_knn)
performance_diff <- generate_validation_results(fit_knn, roc_results)

# Display final results
display_model_summary(fit_knn, roc_results, performance_diff)

# Final artifacts summary
cat("\n=== kNN Artifacts ===\n")
if (model_uploaded) {
  cat("📥 Model uploaded from R environment:\n")
} else if (model_loaded) {
  cat("📁 Model loaded from storage:\n")
} else {
  cat("🔧 Model trained and saved:\n")
}
cat("   -", model_path, "(final model)\n")
if (file.exists(tuning_plot_path)) {
  cat("   -", tuning_plot_path, "(parameter tuning plot)\n")
}
cat("   -", roc_plot_path, "(training ROC curve)\n")
cat("   -", validation_roc_plot_path, "(validation ROC curve)\n")
cat("   -", validation_results_path, "(comprehensive validation results)\n")

# Final success message
cat("\n✅ kNN model processing completed successfully!\n")
if (model_uploaded) {
  cat("📥 Model was uploaded from R environment\n")
} else if (model_loaded) {
  cat("📁 Model was loaded from existing file\n")
} else {
  cat("🔧 Model was trained and saved\n")
}
cat("📊 Best parameter: k =", fit_knn$bestTune$k, "\n")
if (!is.null(fit_knn$results)) {
  cat("🎯 Cross-validation AUC:", round(max(fit_knn$results$ROC), 4), "\n")
}
cat("🎯 Training AUC:", round(roc_results$auc_train, 4), "\n")
cat("🎯 Validation AUC:", round(roc_results$auc_validation, 4), "\n")
cat("📈 Performance Difference:", round(performance_diff, 4), "\n")
cat("📁 Data split: 80% training, 20% validation\n")

# Skip feature importance calculation for kNN (not supported)
cat("\nNote: kNN does not provide built-in feature importance like tree-based models.\n")
cat("This is expected behavior for distance-based algorithms.\n")

# Display results for all attempted k values if available
if (!is.null(fit_knn$results)) {
  cat("\n=== Detailed Tuning Results ===\n")
  print(fit_knn$results)
}

# Helper function for string concatenation
`%+%` <- function(a, b) paste0(a, b)

cat("\n" %+% 
      "🎉 kNN analysis complete!\n" %+%
      "📁 Files generated:\n" %+%
      "   • " %+% model_path %+% " (trained model)\n" %+%
      "   • " %+% tuning_plot_path %+% " (parameter tuning visualization)\n" %+%
      "   • " %+% roc_plot_path %+% " (training ROC curve)\n" %+%
      "   • " %+% validation_roc_plot_path %+% " (validation ROC curve)\n" %+%
      "   • " %+% validation_results_path %+% " (comprehensive validation results)\n")

# ============================================
# Part 2.5 - Model 3: Naive Bayes
# Steps: Split 80/20 -> CV -> validation -> summary -> save artifacts
# ============================================

# Load required packages
req_pkgs <- c("caret", "naivebayes", "ggplot2", "dplyr", "pROC")
to_install <- setdiff(req_pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

suppressPackageStartupMessages({
  library(caret)
  library(naivebayes)
  library(ggplot2)
  library(dplyr)
  library(pROC)
})

set.seed(6104) # Consistent with previous steps

# Input validation
stopifnot(exists("train_stratsample_smote"))

# Ensure target variable format is correct
train_stratsample_smote$is_fraud <- factor(
  ifelse(tolower(as.character(train_stratsample_smote$is_fraud)) %in% 
           c("1","yes","y","true","t"), "yes", "no"),
  levels = c("no", "yes")
)

# ============================================
# NEW: Split data into 80% training and 20% validation
# ============================================
cat("=== Splitting Data into 80% Training and 20% Validation ===\n")

set.seed(6104)  # For reproducibility
train_index <- createDataPartition(train_stratsample_smote$is_fraud, 
                                   p = 0.8, 
                                   list = FALSE, 
                                   times = 1)

train_80 <- train_stratsample_smote[train_index, ]
validation_20 <- train_stratsample_smote[-train_index, ]

cat(sprintf("Training set (80%%): %d observations\n", nrow(train_80)))
cat(sprintf("Validation set (20%%): %d observations\n", nrow(validation_20)))
cat(sprintf("Class distribution in training set:\n"))
print(table(train_80$is_fraud))
cat(sprintf("Class distribution in validation set:\n"))
print(table(validation_20$is_fraud))

# Define file paths
model_path <- "models/model3_nb_final.rds"
tuning_plot_path <- "models/model3_nb_tuning_plot.pdf"
roc_plot_path <- "models/model3_nb_roc_curve.pdf"
validation_roc_plot_path <- "models/model3_nb_validation_roc.pdf"
validation_results_path <- "models/model3_nb_validation_results.pdf"

# Create models directory if it doesn't exist
dir.create("models", showWarnings = FALSE)

# Check ctrl
if (!exists("ctrl")) {
  cat("Note: ctrl not found, redefining trainControl...\n")
  ctrl <- trainControl(
    method = "cv", 
    number = 10,
    classProbs = TRUE, 
    summaryFunction = twoClassSummary,
    savePredictions = "final"
  )
} else {
  cat("Using existing ctrl object...\n")
}

cat("=== Naive Bayes Model Smart Loading/Training ===\n")

# ============================================
# NEW: Check if model is already uploaded in R environment
# ============================================
model_uploaded <- FALSE
if (exists("uploaded_nb_model")) {
  cat("🔍 Checking for uploaded model in R environment...\n")
  if (inherits(uploaded_nb_model, "train")) {
    cat("✅ Found uploaded Naive Bayes model in R environment\n")
    
    # Validate uploaded model structure
    model_vars <- all.vars(uploaded_nb_model$terms)
    current_vars <- names(train_80)
    missing_vars <- setdiff(model_vars, current_vars)
    
    if (length(missing_vars) == 0) {
      fit_nb <- uploaded_nb_model
      model_uploaded <- TRUE
      
      # Extract best parameters from uploaded model
      best_params <- fit_nb$bestTune
      cat("✅ Using uploaded model with parameters:\n")
      print(best_params)
      cat("💾 Saving uploaded model to file for future use...\n")
      saveRDS(fit_nb, file = model_path)
    } else {
      cat("⚠️ Uploaded model has incompatible variables:", paste(missing_vars, collapse = ", "), "\n")
      cat("🔄 Ignoring uploaded model due to feature mismatch...\n")
    }
  } else {
    cat("⚠️ Uploaded object is not a valid caret train model\n")
  }
}

# Function to train Naive Bayes model
train_naive_bayes <- function() {
  cat("Training Naive Bayes model...\n")
  
  # Define explicit parameter grid
  nb_grid <- expand.grid(
    laplace = c(0, 0.5, 1, 2, 5),      # Laplace smoothing parameter
    usekernel = c(TRUE, FALSE),         # Whether to use kernel density estimation
    adjust = 1                          # Bandwidth adjustment parameter (fixed at 1 to simplify)
  )
  
  fit_nb <- train(
    is_fraud ~ .,
    data = train_80,  # Use 80% training data
    method = "naive_bayes",
    trControl = ctrl,
    tuneGrid = nb_grid,                # Use explicit parameter grid
    metric = "ROC"
  )
  
  return(fit_nb)
}

# 1) Try to load pre-trained model; train if not exists and no uploaded model
model_loaded <- FALSE

if (!model_uploaded && file.exists(model_path)) {
  cat("📁 Loading pre-trained Naive Bayes model...\n")
  
  tryCatch({
    fit_nb <- readRDS(model_path)
    
    # Validate loaded model structure
    if (inherits(fit_nb, "train")) {
      cat("✅ Naive Bayes model successfully loaded\n")
      
      # === CHECK FOR MISSING VARIABLES ===
      model_vars <- all.vars(fit_nb$terms)
      current_vars <- names(train_80)
      missing_vars <- setdiff(model_vars, current_vars)
      
      if (length(missing_vars) > 0) {
        cat("⚠️ Missing variables in current data:", paste(missing_vars, collapse = ", "), "\n")
        cat("🔄 Forcing model retraining due to feature mismatch...\n")
        model_loaded <- FALSE
        
        # Remove the problematic model file to prevent future issues
        if (file.exists(model_path)) {
          file.remove(model_path)
          cat("🗑️ Removed incompatible model file:", model_path, "\n")
        }
      } else {
        model_loaded <- TRUE
        
        # Extract best parameters from loaded model
        best_params <- fit_nb$bestTune
        cat("Best parameters from loaded model:\n")
        print(best_params)
      }
    } else {
      cat("⚠️ Warning: Loaded model has incorrect structure. Retraining...\n")
      model_loaded <- FALSE
    }
  }, error = function(e) {
    cat("❌ Error loading model:", e$message, "\n")
    cat("🔄 Retraining model...\n")
    model_loaded <- FALSE
  })
} else if (!model_uploaded) {
  cat("📭 No saved Naive Bayes model found\n")
  model_loaded <- FALSE
}

# 2) Train model if not uploaded and not loaded successfully
if (!model_uploaded && !model_loaded) {
  cat("🔧 Starting Naive Bayes model training...\n")
  fit_nb <- train_naive_bayes()
  
  # Save trained model
  saveRDS(fit_nb, file = model_path)
  cat("💾 Naive Bayes model saved:", model_path, "\n")
} else if (model_uploaded) {
  cat("📥 Using uploaded model, skipping training phase\n")
} else if (model_loaded) {
  cat("📁 Using pre-loaded model, skipping training phase\n")
}

# Function to generate tuning plot
generate_tuning_plot <- function(fit_nb) {
  if (!exists("fit_nb") || is.null(fit_nb$results)) {
    cat("⚠️ Cannot generate tuning plot: model results not available\n")
    return(NULL)
  }
  
  tuning_plot <- ggplot(fit_nb) +
    ggtitle("Naive Bayes Parameter Tuning - Laplace Smoothing vs AUC") +
    xlab("Laplace Smoothing Parameter") +
    ylab("AUC (Cross-Validation)") +
    theme_minimal()
  
  ggsave(tuning_plot_path, tuning_plot, width = 8, height = 6)
  cat("📊 Naive Bayes parameter tuning plot saved:", tuning_plot_path, "\n")
  
  return(tuning_plot)
}

# Function to generate ROC curves for both training and validation
generate_roc_curves <- function(fit_nb) {
  cat("\n=== Generating ROC Curves for Naive Bayes Model ===\n")
  
  # Training set ROC
  nb_probs_train <- predict(fit_nb, newdata = train_80, type = "prob")
  nb_roc_train <- roc(response = train_80$is_fraud, 
                      predictor = nb_probs_train[, "yes"])
  nb_auc_train <- auc(nb_roc_train)
  cat(sprintf("Naive Bayes Training AUC: %.4f\n", nb_auc_train))
  
  # Create training ROC curve plot
  nb_roc_plot_train <- ggroc(nb_roc_train, alpha = 0.8, color = "green", linewidth = 1.2) +
    geom_abline(intercept = 1, slope = 1, linetype = "dashed", alpha = 0.5) +
    labs(title = "Naive Bayes - ROC Curve (Training Set)",
         subtitle = sprintf("AUC = %.4f", nb_auc_train),
         x = "Specificity",
         y = "Sensitivity") +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 12),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_line(color = "grey95")
    ) +
    coord_equal()
  
  # Save training ROC plot
  ggsave(roc_plot_path, nb_roc_plot_train, width = 8, height = 6)
  cat("📊 Training ROC curve saved:", roc_plot_path, "\n")
  
  # ============================================
  # NEW: Validation set ROC
  # ============================================
  nb_probs_validation <- predict(fit_nb, newdata = validation_20, type = "prob")
  nb_roc_validation <- roc(response = validation_20$is_fraud, 
                           predictor = nb_probs_validation[, "yes"])
  nb_auc_validation <- auc(nb_roc_validation)
  cat(sprintf("Naive Bayes Validation AUC: %.4f\n", nb_auc_validation))
  
  # Create validation ROC curve plot
  nb_roc_plot_validation <- ggroc(nb_roc_validation, alpha = 0.8, color = "blue", linewidth = 1.2) +
    geom_abline(intercept = 1, slope = 1, linetype = "dashed", alpha = 0.5) +
    labs(title = "Naive Bayes - ROC Curve (Validation Set)",
         subtitle = sprintf("AUC = %.4f", nb_auc_validation),
         x = "Specificity",
         y = "Sensitivity") +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 12),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_line(color = "grey95")
    ) +
    coord_equal()
  
  # Save validation ROC plot
  ggsave(validation_roc_plot_path, nb_roc_plot_validation, width = 8, height = 6)
  cat("📊 Validation ROC curve saved:", validation_roc_plot_path, "\n")
  
  # Display AUC confidence intervals
  nb_ci_auc_train <- ci.auc(nb_roc_train)
  nb_ci_auc_validation <- ci.auc(nb_roc_validation)
  
  cat(sprintf("Training AUC 95%% Confidence Interval: [%.4f, %.4f]\n", 
              nb_ci_auc_train[1], nb_ci_auc_train[3]))
  cat(sprintf("Validation AUC 95%% Confidence Interval: [%.4f, %.4f]\n", 
              nb_ci_auc_validation[1], nb_ci_auc_validation[3]))
  
  return(list(
    roc_plot_train = nb_roc_plot_train, 
    auc_train = nb_auc_train, 
    ci_auc_train = nb_ci_auc_train,
    roc_plot_validation = nb_roc_plot_validation,
    auc_validation = nb_auc_validation,
    ci_auc_validation = nb_ci_auc_validation,
    roc_train = nb_roc_train,           # 新增：返回 ROC 对象
    roc_validation = nb_roc_validation   # 新增：返回 ROC 对象
  ))
}

# ============================================
# NEW: Generate comprehensive validation results PDF
# ============================================
generate_validation_results <- function(fit_nb, roc_results) {
  cat("\n=== Generating Comprehensive Validation Results PDF ===\n")
  
  pdf(file = validation_results_path, width = 11, height = 8.5)
  
  # Page 1: Validation Performance Summary
  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)
  plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), 
       xlab = "", ylab = "", axes = FALSE, main = "Naive Bayes Validation Results")
  text(0.5, 0.9, "Naive Bayes Model Validation Summary", cex = 1.5, font = 2)
  
  # Model source information
  model_source <- if (model_uploaded) {
    "Uploaded Model"
  } else if (model_loaded) {
    "Pre-loaded Model"
  } else {
    "Newly Trained Model"
  }
  text(0.1, 0.85, sprintf("Model Source: %s", model_source), pos = 4, cex = 1.2)
  
  # Performance metrics
  text(0.1, 0.7, sprintf("Training Set Size: %d observations", nrow(train_80)), pos = 4, cex = 1.2)
  text(0.1, 0.65, sprintf("Validation Set Size: %d observations", nrow(validation_20)), pos = 4, cex = 1.2)
  text(0.1, 0.6, sprintf("Training AUC: %.4f", roc_results$auc_train), pos = 4, cex = 1.2)
  text(0.1, 0.55, sprintf("Validation AUC: %.4f", roc_results$auc_validation), pos = 4, cex = 1.2)
  
  # Best parameters
  best_params <- fit_nb$bestTune
  param_text <- paste(names(best_params), best_params, sep = " = ", collapse = ", ")
  text(0.1, 0.5, sprintf("Best Parameters: %s", param_text), pos = 4, cex = 1.1)
  
  # Performance comparison
  performance_diff <- roc_results$auc_train - roc_results$auc_validation
  if (performance_diff > 0.05) {
    performance_note <- "⚠️ Significant overfitting detected"
  } else if (performance_diff > 0.02) {
    performance_note <- "ℹ️ Moderate overfitting detected"
  } else {
    performance_note <- "✅ Good generalization performance"
  }
  
  text(0.1, 0.4, sprintf("Performance Difference (Train - Val): %.4f", performance_diff), pos = 4, cex = 1.2)
  text(0.1, 0.35, performance_note, pos = 4, cex = 1.2, col = ifelse(performance_diff > 0.05, "red", "darkgreen"))
  
  # Confidence intervals
  text(0.1, 0.25, sprintf("Training AUC 95%% CI: [%.4f, %.4f]", 
                          roc_results$ci_auc_train[1], roc_results$ci_auc_train[3]), pos = 4, cex = 1.1)
  text(0.1, 0.2, sprintf("Validation AUC 95%% CI: [%.4f, %.4f]", 
                         roc_results$ci_auc_validation[1], roc_results$ci_auc_validation[3]), pos = 4, cex = 1.1)
  
  # Page 2: Side-by-side ROC curves
  par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
  # Training ROC - 使用从 roc_results 传递过来的 ROC 对象
  plot(roc_results$roc_train, main = "Training ROC Curve", col = "green", lwd = 2)
  text(0.5, 0.3, sprintf("AUC = %.4f", roc_results$auc_train), cex = 1.2)
  # Validation ROC - 使用从 roc_results 传递过来的 ROC 对象
  plot(roc_results$roc_validation, main = "Validation ROC Curve", col = "blue", lwd = 2)
  text(0.5, 0.3, sprintf("AUC = %.4f", roc_results$auc_validation), cex = 1.2)
  
  # Page 3: Parameter importance (if available in results)
  if (!is.null(fit_nb$results) && nrow(fit_nb$results) > 0) {
    par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)
    
    # Create a simple bar plot showing AUC by Laplace parameter
    if ("laplace" %in% names(fit_nb$results)) {
      laplace_auc <- aggregate(ROC ~ laplace, data = fit_nb$results, mean)
      barplot(laplace_auc$ROC, names.arg = laplace_auc$laplace, 
              col = "steelblue", main = "AUC by Laplace Smoothing Parameter",
              xlab = "Laplace Smoothing", ylab = "AUC")
    }
  }
  
  dev.off()
  cat("✅ Comprehensive validation results saved:", validation_results_path, "\n")
  
  return(performance_diff)
}

# Function to display model summary
display_model_summary <- function(fit_nb, roc_results, performance_diff) {
  cat("\n=== Naive Bayes Model Summary ===\n")
  cat("Best parameters:\n")
  print(fit_nb$bestTune)
  
  if (!is.null(fit_nb$results)) {
    cv_auc <- max(fit_nb$results$ROC, na.rm = TRUE)
    cat("Cross-validation AUC:", round(cv_auc, 4), "\n")
    
    # Sensitivity & Specificity
    best_index <- which.max(fit_nb$results$ROC)
    if (length(best_index) > 0) {
      cat("Sensitivity:", round(fit_nb$results$Sens[best_index], 4), "\n")
      cat("Specificity:", round(fit_nb$results$Spec[best_index], 4), "\n")
    }
  }
  
  cat("Number of features used:", ncol(train_80) - 1, "\n")
  cat("Training AUC:", round(roc_results$auc_train, 4), "\n")
  cat("Validation AUC:", round(roc_results$auc_validation, 4), "\n")
  cat("Performance Difference:", round(performance_diff, 4), "\n")
  cat(sprintf("Training AUC 95%% CI: [%.4f, %.4f]\n", 
              roc_results$ci_auc_train[1], roc_results$ci_auc_train[3]))
  cat(sprintf("Validation AUC 95%% CI: [%.4f, %.4f]\n", 
              roc_results$ci_auc_validation[1], roc_results$ci_auc_validation[3]))
}

# Main execution logic
# Generate tuning plot (only if we trained the model or have CV results)
if ((!model_uploaded && !model_loaded) || (exists("fit_nb") && !is.null(fit_nb$results))) {
  generate_tuning_plot(fit_nb)
} else {
  cat("📊 Parameter tuning plot not available for uploaded/loaded model\n")
}

# Generate ROC curves and validation results
roc_results <- generate_roc_curves(fit_nb)
performance_diff <- generate_validation_results(fit_nb, roc_results)

# Display final results
display_model_summary(fit_nb, roc_results, performance_diff)

# Final artifacts summary
cat("\n=== Naive Bayes Artifacts ===\n")
if (model_uploaded) {
  cat("📥 Model uploaded from R environment:\n")
} else if (model_loaded) {
  cat("📁 Model loaded from storage:\n")
} else {
  cat("🔧 Model trained and saved:\n")
}
cat("   -", model_path, "(final model)\n")
if (file.exists(tuning_plot_path)) {
  cat("   -", tuning_plot_path, "(parameter tuning plot)\n")
}
cat("   -", roc_plot_path, "(training ROC curve)\n")
cat("   -", validation_roc_plot_path, "(validation ROC curve)\n")
cat("   -", validation_results_path, "(comprehensive validation results)\n")

# Final success message
cat("\n✅ Naive Bayes model processing completed successfully!\n")
if (model_uploaded) {
  cat("📥 Model was uploaded from R environment\n")
} else if (model_loaded) {
  cat("📁 Model was loaded from existing file\n")
} else {
  cat("🔧 Model was trained and saved\n")
}
cat("📊 Best parameters:", paste(names(fit_nb$bestTune), fit_nb$bestTune, sep = " = ", collapse = ", "), "\n")
if (!is.null(fit_nb$results)) {
  cat("🎯 Cross-validation AUC:", round(max(fit_nb$results$ROC), 4), "\n")
}
cat("🎯 Training AUC:", round(roc_results$auc_train, 4), "\n")
cat("🎯 Validation AUC:", round(roc_results$auc_validation, 4), "\n")
cat("📈 Performance Difference:", round(performance_diff, 4), "\n")
cat("📁 Data split: 80% training, 20% validation\n")
cat("ℹ️  Feature importance not available for Naive Bayes (this is normal)\n")

# Display all results from tuning if available
if (!is.null(fit_nb$results)) {
  cat("\n=== Detailed Tuning Results ===\n")
  print(fit_nb$results)
}

# Helper function for string concatenation
`%+%` <- function(a, b) paste0(a, b)

cat("\n" %+% 
      "🎉 Naive Bayes analysis complete!\n" %+%
      "📁 Files generated:\n" %+%
      "   • " %+% model_path %+% " (trained model)\n" %+%
      "   • " %+% roc_plot_path %+% " (training ROC curve)\n" %+%
      "   • " %+% validation_roc_plot_path %+% " (validation ROC curve)\n" %+%
      "   • " %+% validation_results_path %+% " (comprehensive validation results)\n")

# ============================================
# Part 2.4 - Model Comparison on Validation Set
# Compare: Decision Tree vs kNN vs Naive Bayes
# Metrics: AUC, Recall, F1-score, Precision, Confusion Matrix
# ============================================

# Load required packages
req_pkgs <- c("caret", "ggplot2", "pROC", "dplyr", "tidyr", "gridExtra", "pander", "grid")
to_install <- setdiff(req_pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

suppressPackageStartupMessages({
  library(caret)
  library(ggplot2)
  library(pROC)
  library(dplyr)
  library(tidyr)
  library(gridExtra)
  library(pander)
  library(grid)
})

set.seed(6104)

# Input validation
stopifnot(exists("validation_20"), exists("train_80"))

cat("=== Comprehensive Model Comparison on Validation Set ===\n")

# Define file paths for comparison results
comparison_plot_path <- "models/model_comparison_metrics.pdf"
comparison_results_path <- "models/model_comparison_results.pdf"
confusion_matrix_path <- "models/model_confusion_matrices.pdf"
performance_table_path <- "models/model_performance_table.csv"

# Create models directory if it doesn't exist
dir.create("models", showWarnings = FALSE)

# ============================================
# Load or train models (if not already in environment)
# ============================================

# Check if models exist in environment, otherwise load from files
models_list <- list()

# Decision Tree
if (exists("final_tree")) {
  cat("✅ Using existing Decision Tree model from environment\n")
  models_list$Decision_Tree <- final_tree
} else if (file.exists("models/model1_tree_final.rds")) {
  cat("📁 Loading Decision Tree model from file\n")
  models_list$Decision_Tree <- readRDS("models/model1_tree_final.rds")
} else {
  cat("❌ Decision Tree model not found. Please run Decision Tree training first.\n")
  stop("Decision Tree model missing")
}

# kNN
if (exists("fit_knn")) {
  cat("✅ Using existing kNN model from environment\n")
  models_list$kNN <- fit_knn
} else if (file.exists("models/model2_knn_final.rds")) {
  cat("📁 Loading kNN model from file\n")
  models_list$kNN <- readRDS("models/model2_knn_final.rds")
} else {
  cat("❌ kNN model not found. Please run kNN training first.\n")
  stop("kNN model missing")
}

# Naive Bayes
if (exists("fit_nb")) {
  cat("✅ Using existing Naive Bayes model from environment\n")
  models_list$Naive_Bayes <- fit_nb
} else if (file.exists("models/model3_nb_final.rds")) {
  cat("📁 Loading Naive Bayes model from file\n")
  models_list$Naive_Bayes <- readRDS("models/model3_nb_final.rds")
} else {
  cat("❌ Naive Bayes model not found. Please run Naive Bayes training first.\n")
  stop("Naive Bayes model missing")
}

# ============================================
# IMPROVED: Robust function to calculate comprehensive metrics
# ============================================
calculate_metrics_robust <- function(predictions, probabilities, actual, model_name) {
  # Ensure factors have correct levels
  predictions <- factor(predictions, levels = c("no", "yes"))
  actual <- factor(actual, levels = c("no", "yes"))
  
  # Diagnostic information
  cat(sprintf("\n🔍 %s Prediction Diagnostics:\n", model_name))
  cat("Prediction distribution:\n")
  pred_table <- table(predictions)
  print(pred_table)
  cat("Actual distribution:\n")
  actual_table <- table(actual)
  print(actual_table)
  
  # Check if we have predictions for both classes
  has_both_classes <- all(c("no", "yes") %in% levels(predictions)) && 
    all(c("no", "yes") %in% levels(actual))
  
  if (!has_both_classes) {
    cat("⚠️ Warning: Missing one or more class levels in predictions or actual values\n")
  }
  
  # Calculate confusion matrix with error handling
  cm <- tryCatch({
    confusionMatrix(predictions, actual, positive = "yes")
  }, error = function(e) {
    cat("❌ Error in confusion matrix:", e$message, "\n")
    cat("🔄 Creating manual confusion matrix...\n")
    
    # Create manual confusion matrix
    manual_cm <- table(Prediction = predictions, Reference = actual)
    manual_cm <- as.matrix(manual_cm)
    
    # Calculate basic metrics manually if needed
    return(list(
      table = manual_cm,
      overall = c(Accuracy = NA),
      byClass = c(
        Sensitivity = NA,
        Specificity = NA,
        Precision = NA,
        F1 = NA
      )
    ))
  })
  
  # Extract metrics with safety checks
  accuracy <- ifelse("Accuracy" %in% names(cm$overall), 
                     as.numeric(cm$overall["Accuracy"]), NA)
  
  sensitivity <- ifelse("Sensitivity" %in% names(cm$byClass), 
                        as.numeric(cm$byClass["Sensitivity"]), NA)
  specificity <- ifelse("Specificity" %in% names(cm$byClass), 
                        as.numeric(cm$byClass["Specificity"]), NA)
  precision <- ifelse("Precision" %in% names(cm$byClass), 
                      as.numeric(cm$byClass["Precision"]), NA)
  f1 <- ifelse("F1" %in% names(cm$byClass), 
               as.numeric(cm$byClass["F1"]), NA)
  
  # Manual F1 calculation if needed
  if (is.na(f1) && !is.na(precision) && !is.na(sensitivity) && (precision + sensitivity) > 0) {
    f1 <- 2 * (precision * sensitivity) / (precision + sensitivity)
    cat(sprintf("📊 Manually calculated F1: %.4f\n", f1))
  }
  
  # Calculate AUC with error handling
  auc_val <- NA
  roc_obj <- NULL
  tryCatch({
    roc_obj <- roc(response = actual, predictor = probabilities[, "yes"])
    auc_val <- auc(roc_obj)
    cat(sprintf("📊 %s AUC: %.4f\n", model_name, auc_val))
  }, error = function(e) {
    cat("❌ Error calculating AUC:", e$message, "\n")
  })
  
  # Calculate balanced accuracy
  balanced_accuracy <- ifelse(!is.na(sensitivity) && !is.na(specificity),
                              (sensitivity + specificity) / 2, NA)
  
  return(list(
    model = model_name,
    accuracy = as.numeric(accuracy),
    sensitivity = as.numeric(sensitivity),
    specificity = as.numeric(specificity),
    precision = as.numeric(precision),
    f1 = as.numeric(f1),
    auc = as.numeric(auc_val),
    balanced_accuracy = as.numeric(balanced_accuracy),
    confusion_matrix = cm,
    roc = roc_obj,
    predictions = predictions,
    probabilities = probabilities
  ))
}

# ============================================
# IMPROVED: Function to optimize prediction threshold for Naive Bayes
# ============================================
optimize_threshold <- function(probabilities, actual, model_name) {
  cat(sprintf("\n🎯 Optimizing threshold for %s...\n", model_name))
  
  thresholds <- seq(0.1, 0.9, by = 0.05)
  best_threshold <- 0.5
  best_f1 <- 0
  
  for (threshold in thresholds) {
    temp_predictions <- ifelse(probabilities[, "yes"] > threshold, "yes", "no")
    temp_predictions <- factor(temp_predictions, levels = c("no", "yes"))
    
    # Calculate F1 manually to avoid confusionMatrix errors
    tp <- sum(temp_predictions == "yes" & actual == "yes")
    fp <- sum(temp_predictions == "yes" & actual == "no")
    fn <- sum(temp_predictions == "no" & actual == "yes")
    
    precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
    recall <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
    f1 <- ifelse((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0)
    
    if (f1 > best_f1) {
      best_f1 <- f1
      best_threshold <- threshold
    }
  }
  
  cat(sprintf("✅ Best threshold for %s: %.3f (F1: %.4f)\n", model_name, best_threshold, best_f1))
  return(best_threshold)
}

# ============================================
# Generate predictions and calculate metrics for all models
# ============================================
cat("\n=== Generating Predictions on Validation Set ===\n")

results <- list()

# Decision Tree predictions
cat("🔮 Generating Decision Tree predictions...\n")
dt_predictions <- predict(models_list$Decision_Tree, newdata = validation_20, type = "class")
dt_probabilities <- predict(models_list$Decision_Tree, newdata = validation_20, type = "prob")
results$Decision_Tree <- calculate_metrics_robust(dt_predictions, dt_probabilities, validation_20$is_fraud, "Decision Tree")

# kNN predictions
cat("🔮 Generating kNN predictions...\n")
knn_predictions <- predict(models_list$kNN, newdata = validation_20, type = "raw")
knn_probabilities <- predict(models_list$kNN, newdata = validation_20, type = "prob")
results$kNN <- calculate_metrics_robust(knn_predictions, knn_probabilities, validation_20$is_fraud, "kNN")

# Naive Bayes predictions with threshold optimization
cat("🔮 Generating Naive Bayes predictions...\n")
nb_probabilities <- predict(models_list$Naive_Bayes, newdata = validation_20, type = "prob")

# Get default predictions first
nb_predictions_default <- predict(models_list$Naive_Bayes, newdata = validation_20, type = "raw")

# Check if we need threshold optimization
cat("Naive Bayes default prediction distribution:\n")
print(table(nb_predictions_default))

# If no "yes" predictions or poor distribution, optimize threshold
if (!"yes" %in% levels(nb_predictions_default) || sum(nb_predictions_default == "yes") == 0) {
  cat("🔄 No 'yes' predictions detected. Optimizing threshold...\n")
  optimal_threshold <- optimize_threshold(nb_probabilities, validation_20$is_fraud, "Naive Bayes")
  nb_predictions_optimized <- ifelse(nb_probabilities[, "yes"] > optimal_threshold, "yes", "no")
  nb_predictions_optimized <- factor(nb_predictions_optimized, levels = c("no", "yes"))
  cat("Optimized prediction distribution:\n")
  print(table(nb_predictions_optimized))
  results$Naive_Bayes <- calculate_metrics_robust(nb_predictions_optimized, nb_probabilities, validation_20$is_fraud, "Naive Bayes")
} else {
  # Use default predictions
  results$Naive_Bayes <- calculate_metrics_robust(nb_predictions_default, nb_probabilities, validation_20$is_fraud, "Naive Bayes")
}

# ============================================
# Create comparison data frame with safety checks
# ============================================
create_comparison_df <- function(results) {
  comparison_data <- list()
  
  for (model_name in names(results)) {
    result <- results[[model_name]]
    
    comparison_data[[model_name]] <- data.frame(
      Model = result$model,
      AUC = ifelse(is.null(result$auc) || is.na(result$auc), NA, result$auc),
      Recall = ifelse(is.null(result$sensitivity) || is.na(result$sensitivity), NA, result$sensitivity),
      Precision = ifelse(is.null(result$precision) || is.na(result$precision), NA, result$precision),
      F1_Score = ifelse(is.null(result$f1) || is.na(result$f1), NA, result$f1),
      Accuracy = ifelse(is.null(result$accuracy) || is.na(result$accuracy), NA, result$accuracy),
      Specificity = ifelse(is.null(result$specificity) || is.na(result$specificity), NA, result$specificity),
      Balanced_Accuracy = ifelse(is.null(result$balanced_accuracy) || is.na(result$balanced_accuracy), NA, result$balanced_accuracy),
      stringsAsFactors = FALSE
    )
  }
  
  comparison_df <- do.call(rbind, comparison_data)
  rownames(comparison_df) <- NULL
  
  # Sort by AUC (primary metric), handling NA values
  if (all(!is.na(comparison_df$AUC))) {
    comparison_df <- comparison_df[order(-comparison_df$AUC), ]
  }
  
  return(comparison_df)
}

comparison_df <- create_comparison_df(results)

# Save performance table
write.csv(comparison_df, performance_table_path, row.names = FALSE)
cat("💾 Performance table saved:", performance_table_path, "\n")

cat("\n=== Model Performance Comparison ===\n")
print(comparison_df)

# ============================================
# Create comprehensive comparison visualization
# ============================================
cat("\n=== Generating Comparison Visualizations ===\n")

# 1. Metrics comparison plot (only include models with valid metrics)
valid_models <- comparison_df[complete.cases(comparison_df[, c("AUC", "Recall", "F1_Score")]), ]

if (nrow(valid_models) > 0) {
  metrics_long <- valid_models %>%
    select(Model, AUC, Recall, F1_Score, Accuracy) %>%
    pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")
  
  metrics_plot <- ggplot(metrics_long, aes(x = Model, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = "Model Performance Comparison on Validation Set",
         subtitle = "Higher values are better for all metrics",
         y = "Score", x = "Model") +
    scale_fill_brewer(palette = "Set2") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    geom_text(aes(label = round(Value, 3)), 
              position = position_dodge(width = 0.9), 
              vjust = -0.5, size = 3) +
    ylim(0, 1)
} else {
  metrics_plot <- ggplot() + 
    theme_void() +
    labs(title = "No valid metrics available for visualization")
}

# 2. ROC curves comparison (only include models with valid ROC)
roc_plots <- list()
roc_data_list <- list()

for (model_name in names(results)) {
  if (!is.null(results[[model_name]]$roc) && !is.na(results[[model_name]]$auc)) {
    roc_data_list[[model_name]] <- data.frame(
      Sensitivity = results[[model_name]]$roc$sensitivities,
      Specificity = results[[model_name]]$roc$specificities,
      Model = model_name
    )
  }
}

if (length(roc_data_list) > 0) {
  roc_data <- do.call(rbind, roc_data_list)
  
  roc_plot <- ggplot(roc_data, aes(x = 1 - Specificity, y = Sensitivity, color = Model)) +
    geom_line(size = 1.2) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", alpha = 0.5) +
    labs(title = "ROC Curves Comparison",
         x = "1 - Specificity (False Positive Rate)",
         y = "Sensitivity (True Positive Rate)") +
    scale_color_manual(values = c("Decision_Tree" = "red", 
                                  "kNN" = "blue", 
                                  "Naive_Bayes" = "green")) +
    theme_minimal() +
    theme(legend.position = "bottom") +
    coord_equal()
  
  # Add AUC values to subtitle
  auc_subtitle <- paste(sapply(names(results), function(x) {
    if (!is.null(results[[x]]$roc) && !is.na(results[[x]]$auc)) {
      paste0(x, " AUC:", round(results[[x]]$auc, 4))
    } else {
      NULL
    }
  }), collapse = " | ")
  
  roc_plot <- roc_plot + labs(subtitle = auc_subtitle)
} else {
  roc_plot <- ggplot() + 
    theme_void() +
    labs(title = "No valid ROC curves available for visualization")
}

# 3. Confusion matrices visualization
create_confusion_matrix_plot <- function(result, model_name) {
  if (!is.null(result$confusion_matrix) && !is.null(result$confusion_matrix$table)) {
    cm_data <- as.data.frame(result$confusion_matrix$table)
    
    plot <- ggplot(cm_data, aes(x = Reference, y = Prediction, fill = Freq)) +
      geom_tile(color = "white") +
      geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
      scale_fill_gradient(low = "blue", high = "red") +
      labs(title = paste("Confusion Matrix -", model_name),
           x = "Actual",
           y = "Predicted") +
      theme_minimal() +
      theme(legend.position = "none",
            plot.title = element_text(hjust = 0.5, face = "bold"))
    
    return(plot)
  } else {
    # Return empty plot if no confusion matrix
    return(ggplot() + 
             theme_void() +
             labs(title = paste("No confusion matrix for", model_name)))
  }
}

cm_plots <- lapply(names(results), function(x) {
  create_confusion_matrix_plot(results[[x]], x)
})
names(cm_plots) <- names(results)

# 4. Create a nice table for PDF
create_performance_table <- function(df) {
  # Round all numeric columns to 4 decimal places
  df_numeric <- df %>% 
    mutate(across(where(is.numeric), ~round(., 4)))
  
  # Create table grob
  table_grob <- tableGrob(df_numeric, 
                          theme = ttheme_minimal(
                            base_size = 10,
                            padding = unit(c(4, 4), "mm")
                          ))
  
  return(table_grob)
}

performance_table <- create_performance_table(comparison_df)

# ============================================
# Save comparison results
# ============================================

# Save metrics comparison plot
ggsave(comparison_plot_path, metrics_plot, width = 12, height = 8)
cat("📊 Metrics comparison plot saved:", comparison_plot_path, "\n")

# Save comprehensive PDF report
pdf(file = comparison_results_path, width = 14, height = 10)

# Page 1: Title and performance summary
grid.text("Model Performance Comparison Summary", 
          x = 0.5, y = 0.95, 
          gp = gpar(fontsize = 18, fontface = "bold"))
grid.text(paste("Validation Set Size:", nrow(validation_20), "observations"),
          x = 0.5, y = 0.90,
          gp = gpar(fontsize = 12))
grid.text(paste("Date:", Sys.Date()),
          x = 0.5, y = 0.86,
          gp = gpar(fontsize = 10))
grid.newpage()

# Page 2: Performance table
grid.text("Model Performance Metrics", 
          x = 0.5, y = 0.95, 
          gp = gpar(fontsize = 16, fontface = "bold"))
grid.draw(performance_table)
grid.newpage()

# Page 3: Metrics comparison
print(metrics_plot)
grid.newpage()

# Page 4: ROC curves
print(roc_plot)
grid.newpage()

# Page 5: Confusion matrices
grid.text("Confusion Matrices Comparison", 
          x = 0.5, y = 0.97, 
          gp = gpar(fontsize = 16, fontface = "bold"))
pushViewport(viewport(layout = grid.layout(2, 2)))
for (i in 1:min(4, length(cm_plots))) {
  row <- ceiling(i / 2)
  col <- ifelse(i %% 2 == 1, 1, 2)
  print(cm_plots[[i]], vp = viewport(layout.pos.row = row, layout.pos.col = col))
}
popViewport()

dev.off()
cat("📊 Comprehensive comparison report saved:", comparison_results_path, "\n")

# Save separate confusion matrices PDF
pdf(file = confusion_matrix_path, width = 12, height = 8)
grid.arrange(grobs = cm_plots, ncol = 2, 
             top = textGrob("Confusion Matrices - All Models", 
                            gp = gpar(fontsize = 16, fontface = "bold")))
dev.off()
cat("📊 Confusion matrices saved:", confusion_matrix_path, "\n")

# ============================================
# Detailed performance analysis
# ============================================
cat("\n=== Detailed Performance Analysis ===\n")

# Find best model for each metric (handle NA values)
find_best_model <- function(metric_vector, metric_name) {
  valid_values <- metric_vector[!is.na(metric_vector)]
  if (length(valid_values) > 0) {
    best_index <- which.max(valid_values)
    best_model <- names(valid_values)[best_index]
    best_value <- valid_values[best_index]
    return(list(model = best_model, value = best_value))
  } else {
    return(list(model = "None", value = NA))
  }
}

# Create named vectors for each metric
auc_values <- setNames(comparison_df$AUC, comparison_df$Model)
f1_values <- setNames(comparison_df$F1_Score, comparison_df$Model)
recall_values <- setNames(comparison_df$Recall, comparison_df$Model)
balanced_values <- setNames(comparison_df$Balanced_Accuracy, comparison_df$Model)

best_auc <- find_best_model(auc_values, "AUC")
best_f1 <- find_best_model(f1_values, "F1")
best_recall <- find_best_model(recall_values, "Recall")
best_balanced <- find_best_model(balanced_values, "Balanced Accuracy")

cat("🏆 Performance Champions:\n")
cat(sprintf("• Best AUC: %s (%.4f)\n", best_auc$model, best_auc$value))
cat(sprintf("• Best F1-Score: %s (%.4f)\n", best_f1$model, best_f1$value))
cat(sprintf("• Best Recall: %s (%.4f)\n", best_recall$model, best_recall$value))
cat(sprintf("• Best Balanced Accuracy: %s (%.4f)\n", best_balanced$model, best_balanced$value))

# Performance differences
cat("\n📈 Performance Analysis:\n")
valid_auc <- auc_values[!is.na(auc_values)]
valid_f1 <- f1_values[!is.na(f1_values)]

if (length(valid_auc) > 1) {
  auc_range <- range(valid_auc)
  cat(sprintf("• AUC range: %.4f (min) to %.4f (max) | Difference: %.4f\n", 
              auc_range[1], auc_range[2], auc_range[2] - auc_range[1]))
}

if (length(valid_f1) > 1) {
  f1_range <- range(valid_f1)
  cat(sprintf("• F1-Score range: %.4f (min) to %.4f (max) | Difference: %.4f\n", 
              f1_range[1], f1_range[2], f1_range[2] - f1_range[1]))
}

# ============================================
# Business context interpretation
# ============================================
cat("\n=== Business Context Interpretation ===\n")

cat("💼 Fraud Detection Context:\n")
cat("• High Recall = Catch most fraudulent transactions\n")
cat("• High Precision = Minimize false alarms (legitimate transactions flagged as fraud)\n")
cat("• F1-Score = Balanced measure considering both recall and precision\n")
cat("• AUC = Overall discrimination ability\n\n")

# Calculate cost-based metrics (simplified)
cat("💰 Cost-Based Considerations:\n")
for (model_name in names(results)) {
  if (!is.null(results[[model_name]]$confusion_matrix) && 
      !is.null(results[[model_name]]$confusion_matrix$table)) {
    cm <- results[[model_name]]$confusion_matrix$table
    false_positives <- ifelse("yes" %in% rownames(cm) && "no" %in% colnames(cm), 
                              cm["yes", "no"], 0)
    false_negatives <- ifelse("no" %in% rownames(cm) && "yes" %in% colnames(cm), 
                              cm["no", "yes"], 0)
    
    cat(sprintf("• %s: %d false positives, %d false negatives\n", 
                model_name, false_positives, false_negatives))
  }
}

# Recommend best model based on business priorities
cat("\n🎯 RECOMMENDATION STRATEGY:\n")
if (best_auc$model == best_f1$model && best_auc$model != "None") {
  cat(sprintf("• OVERALL: %s performs best with AUC=%.4f and F1=%.4f\n", 
              best_auc$model, best_auc$value, best_f1$value))
} else {
  if (best_recall$model != "None") {
    cat(sprintf("• For maximum fraud detection (Recall): %s (Recall=%.4f)\n", best_recall$model, best_recall$value))
  }
  if (best_f1$model != "None") {
    cat(sprintf("• For balanced performance (F1): %s (F1=%.4f)\n", best_f1$model, best_f1$value))
  }
  if (best_auc$model != "None") {
    cat(sprintf("• For overall discrimination (AUC): %s (AUC=%.4f)\n", best_auc$model, best_auc$value))
  }
}

# ============================================
# Final summary
# ============================================
cat("\n=== Model Comparison Complete ===\n")
cat("📁 Generated Files:\n")
cat("•", comparison_plot_path, "- Metrics comparison visualization\n")
cat("•", comparison_results_path, "- Comprehensive comparison report\n")
cat("•", confusion_matrix_path, "- Confusion matrices\n")
cat("•", performance_table_path, "- Performance metrics table\n")
cat("📊 Validation Set Size:", nrow(validation_20), "observations\n")
cat("🎯 Primary Evaluation Metric: AUC (Area Under ROC Curve)\n")
cat("📈 Secondary Metrics: Recall, F1-Score, Precision, Accuracy\n")

# Display final comparison table
cat("\n=== Final Model Ranking ===\n")
print(comparison_df)

cat("\n✅ Model comparison completed successfully!\n")
cat("💡 Tip: Consider your business priorities when choosing the final model:\n")
cat("   - High Recall: Important if missing fraud is costly\n")
cat("   - High Precision: Important if false alarms are costly\n")
cat("   - Balanced (F1): Good compromise for most scenarios\n")

# ============================================
# Part 2.5 - Final Model Evaluation on Test Set
# Evaluate the best model (Decision Tree) on unseen test data
# ============================================

# Load required packages
req_pkgs <- c("caret", "ggplot2", "pROC", "dplyr", "gridExtra", "pander")
to_install <- setdiff(req_pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

suppressPackageStartupMessages({
  library(caret)
  library(ggplot2)
  library(pROC)
  library(dplyr)
  library(gridExtra)
  library(pander)
})

set.seed(6104)

cat("=== Final Model Evaluation on Test Set ===\n")

# Input validation
stopifnot(exists("test_stratsample_50k_fe"))
if (!exists("final_tree")) {
  if (file.exists("models/model1_tree_final.rds")) {
    cat("📁 Loading Decision Tree model from file...\n")
    final_tree <- readRDS("models/model1_tree_final.rds")
  } else {
    stop("❌ Decision Tree model not found. Please run model training first.")
  }
}

# Define file paths for final evaluation
final_test_results_path <- "models/final_tree_test_results.pdf"
final_test_roc_path <- "models/final_tree_test_roc.pdf"
final_test_confusion_path <- "models/final_tree_test_confusion.pdf"
final_test_summary_path <- "models/final_tree_test_summary.csv"

# Create models directory if it doesn't exist
dir.create("models", showWarnings = FALSE)

# ============================================
# Data Preparation and Validation
# ============================================

cat("🔍 Validating test data structure...\n")

# Ensure test data has the same structure as training data
# Check if target variable exists and has correct format
if (!"is_fraud" %in% names(test_stratsample_50k_fe)) {
  stop("❌ Target variable 'is_fraud' not found in test data")
}

# Convert target variable to factor with correct levels
test_stratsample_50k_fe$is_fraud <- factor(
  ifelse(tolower(as.character(test_stratsample_50k_fe$is_fraud)) %in% 
           c("1","yes","y","true","t"), "yes", "no"),
  levels = c("no", "yes")
)

cat(sprintf("Test set size: %d observations\n", nrow(test_stratsample_50k_fe)))
cat("Class distribution in test set:\n")
test_distribution <- table(test_stratsample_50k_fe$is_fraud)
print(test_distribution)
cat(sprintf("Fraud rate: %.4f%%\n", 
            (test_distribution["yes"] / sum(test_distribution)) * 100))

# Check feature compatibility
train_features <- names(final_tree$variable.importance)
test_features <- names(test_stratsample_50k_fe)
missing_features <- setdiff(train_features, test_features)

if (length(missing_features) > 0) {
  cat("⚠️ Warning: Missing features in test data:", paste(missing_features, collapse = ", "), "\n")
  cat("🔄 Attempting to proceed with available features...\n")
} else {
  cat("✅ All features are available in test data\n")
}

# ============================================
# Generate Predictions on Test Set
# ============================================

cat("\n🎯 Generating predictions on test set...\n")

# Generate class predictions
test_predictions <- predict(final_tree, newdata = test_stratsample_50k_fe, type = "class")

# Generate probability predictions
test_probabilities <- predict(final_tree, newdata = test_stratsample_50k_fe, type = "prob")

cat("Prediction distribution:\n")
print(table(test_predictions))

# ============================================
# Calculate Comprehensive Performance Metrics
# ============================================

cat("\n📊 Calculating performance metrics...\n")

# Create confusion matrix
test_cm <- confusionMatrix(test_predictions, test_stratsample_50k_fe$is_fraud, positive = "yes")

# Extract key metrics
accuracy <- test_cm$overall["Accuracy"]
sensitivity <- test_cm$byClass["Sensitivity"]  # Recall
specificity <- test_cm$byClass["Specificity"]
precision <- test_cm$byClass["Precision"]
f1 <- test_cm$byClass["F1"]

# Calculate ROC and AUC
test_roc <- roc(response = test_stratsample_50k_fe$is_fraud, 
                predictor = test_probabilities[, "yes"])
test_auc <- auc(test_roc)
test_auc_ci <- ci.auc(test_roc)

# Calculate balanced accuracy
balanced_accuracy <- (sensitivity + specificity) / 2

# Calculate additional business metrics
confusion_table <- test_cm$table
false_positives <- confusion_table["yes", "no"]  # Type I error
false_negatives <- confusion_table["no", "yes"]  # Type II error
true_positives <- confusion_table["yes", "yes"]
true_negatives <- confusion_table["no", "no"]

# Calculate detection rate and false alarm rate
detection_rate <- true_positives / (true_positives + false_negatives)
false_alarm_rate <- false_positives / (false_positives + true_negatives)

# ============================================
# Create Performance Summary
# ============================================

performance_summary <- data.frame(
  Metric = c(
    "Accuracy", "Sensitivity (Recall)", "Specificity", "Precision",
    "F1-Score", "Balanced Accuracy", "AUC",
    "True Positives", "True Negatives", 
    "False Positives", "False Negatives",
    "Detection Rate", "False Alarm Rate"
  ),
  Value = c(
    round(accuracy, 4),
    round(sensitivity, 4),
    round(specificity, 4),
    round(precision, 4),
    round(f1, 4),
    round(balanced_accuracy, 4),
    round(test_auc, 4),
    true_positives,
    true_negatives,
    false_positives,
    false_negatives,
    round(detection_rate, 4),
    round(false_alarm_rate, 4)
  ),
  Description = c(
    "Overall correctness",
    "Ability to detect fraud (True Positive Rate)",
    "Ability to identify legitimate transactions",
    "Precision in fraud detection",
    "Balance between precision and recall",
    "Average of sensitivity and specificity",
    "Overall classification performance",
    "Correctly identified fraud cases",
    "Correctly identified legitimate transactions",
    "Legitimate transactions incorrectly flagged as fraud",
    "Fraud cases missed",
    "Proportion of fraud cases detected",
    "Proportion of false alarms"
  )
)

# Save performance summary
write.csv(performance_summary, final_test_summary_path, row.names = FALSE)
cat("💾 Performance summary saved:", final_test_summary_path, "\n")

# ============================================
# Create Visualizations
# ============================================

cat("\n🎨 Generating visualizations...\n")

# 1. ROC Curve
roc_plot <- ggroc(test_roc, alpha = 0.8, color = "red", linewidth = 1.5) +
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", alpha = 0.5) +
  labs(title = "Decision Tree - ROC Curve (Test Set)",
       subtitle = sprintf("AUC = %.4f (95%% CI: %.4f - %.4f)", 
                          test_auc, test_auc_ci[1], test_auc_ci[3]),
       x = "Specificity",
       y = "Sensitivity") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_line(color = "grey95")
  ) +
  coord_equal()

# 2. Confusion Matrix Visualization
cm_data <- as.data.frame(test_cm$table)
cm_plot <- ggplot(cm_data, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = Freq), color = "white", size = 8, fontface = "bold") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Confusion Matrix - Decision Tree (Test Set)",
       subtitle = sprintf("Accuracy: %.2f%%", accuracy * 100),
       x = "Actual Class",
       y = "Predicted Class") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    legend.position = "none"
  )

# 3. Feature Importance Plot (if available)
if (!is.null(final_tree$variable.importance)) {
  importance_df <- data.frame(
    Feature = names(final_tree$variable.importance),
    Importance = as.numeric(final_tree$variable.importance)
  ) %>% 
    arrange(desc(Importance)) %>%
    head(10)  # Top 10 features
  
  importance_plot <- ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_col(fill = "steelblue", alpha = 0.8) +
    coord_flip() +
    labs(title = "Top 10 Feature Importance - Decision Tree",
         x = "Features",
         y = "Importance") +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.text.y = element_text(size = 10)
    )
} else {
  importance_plot <- ggplot() + 
    theme_void() +
    labs(title = "Feature Importance Not Available")
}

# ============================================
# Generate Comprehensive Final Report
# ============================================

cat("📋 Generating comprehensive final report...\n")

pdf(file = final_test_results_path, width = 14, height = 10)

# Page 1: Executive Summary
grid.arrange(
  grobs = list(
    grid::textGrob("Final Model Evaluation Report", 
                   gp = grid::gpar(fontsize = 24, fontface = "bold")),
    grid::textGrob("Decision Tree Model Performance on Test Set", 
                   gp = grid::gpar(fontsize = 18)),
    grid::textGrob(paste("Evaluation Date:", Sys.Date()), 
                   gp = grid::gpar(fontsize = 12)),
    grid::textGrob(paste("Test Set Size:", nrow(test_stratsample_50k_fe), "observations"), 
                   gp = grid::gpar(fontsize = 12)),
    grid::textGrob(paste("Fraud Cases:", sum(test_stratsample_50k_fe$is_fraud == "yes"), 
                         "(", round(mean(test_stratsample_50k_fe$is_fraud == "yes") * 100, 2), "%)"), 
                   gp = grid::gpar(fontsize = 12))
  ),
  ncol = 1,
  heights = c(0.2, 0.15, 0.1, 0.1, 0.1)
)

# Page 2: Key Performance Metrics
grid.newpage()
grid.text("Key Performance Metrics", 
          x = 0.5, y = 0.9, 
          gp = gpar(fontsize = 20, fontface = "bold"))

# Create a nice table for key metrics
key_metrics <- performance_summary[1:7, ]
table_grob <- tableGrob(key_metrics, 
                        theme = ttheme_minimal(
                          base_size = 12,
                          padding = unit(c(8, 6), "mm")
                        ))
grid.draw(table_grob)

# Page 3: Confusion Matrix Details
grid.newpage()
grid.text("Confusion Matrix Analysis", 
          x = 0.5, y = 0.95, 
          gp = gpar(fontsize = 20, fontface = "bold"))

pushViewport(viewport(layout = grid.layout(1, 2)))

# Confusion matrix plot
pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 1))
print(cm_plot, vp = viewport(x = 0.5, y = 0.5, width = 0.9, height = 0.9))
popViewport()

# Confusion matrix details
pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 2))
confusion_details <- data.frame(
  Metric = c("True Positives", "True Negatives", 
             "False Positives", "False Negatives",
             "Detection Rate", "False Alarm Rate"),
  Value = c(true_positives, true_negatives,
            false_positives, false_negatives,
            sprintf("%.2f%%", detection_rate * 100),
            sprintf("%.2f%%", false_alarm_rate * 100))
)
details_grob <- tableGrob(confusion_details, 
                          theme = ttheme_minimal(base_size = 10))
grid.draw(details_grob)
popViewport()

popViewport()

# Page 4: ROC Curve and Feature Importance
grid.newpage()
grid.text("Model Performance Visualizations", 
          x = 0.5, y = 0.97, 
          gp = gpar(fontsize = 20, fontface = "bold"))

pushViewport(viewport(layout = grid.layout(1, 2)))

# ROC curve
pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 1))
print(roc_plot, vp = viewport(x = 0.5, y = 0.5, width = 0.9, height = 0.9))
popViewport()

# Feature importance
pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 2))
print(importance_plot, vp = viewport(x = 0.5, y = 0.5, width = 0.9, height = 0.9))
popViewport()

popViewport()

# Page 5: Business Impact Analysis
grid.newpage()
grid.text("Business Impact Analysis", 
          x = 0.5, y = 0.95, 
          gp = gpar(fontsize = 20, fontface = "bold"))

# Calculate business metrics
total_transactions <- nrow(test_stratsample_50k_fe)
fraud_caught <- true_positives
fraud_missed <- false_negatives
false_alarms <- false_positives

business_analysis <- data.frame(
  Metric = c(
    "Total Transactions Analyzed",
    "Fraudulent Transactions Detected",
    "Fraudulent Transactions Missed",
    "False Alarms Generated",
    "Fraud Detection Rate",
    "False Positive Rate",
    "Overall Accuracy"
  ),
  Value = c(
    format(total_transactions, big.mark = ","),
    paste(fraud_caught, sprintf("(%.2f%%)", detection_rate * 100)),
    paste(fraud_missed, sprintf("(%.2f%%)", (false_negatives/(true_positives+false_negatives)) * 100)),
    paste(false_alarms, sprintf("(%.2f%%)", false_alarm_rate * 100)),
    sprintf("%.2f%%", detection_rate * 100),
    sprintf("%.2f%%", false_alarm_rate * 100),
    sprintf("%.2f%%", accuracy * 100)
  ),
  Business_Impact = c(
    "Scale of analysis",
    "Fraud prevention effectiveness",
    "Potential financial losses",
    "Customer inconvenience cost",
    "Key performance indicator",
    "Operational efficiency",
    "Overall system reliability"
  )
)

business_grob <- tableGrob(business_analysis, 
                           theme = ttheme_minimal(
                             base_size = 11,
                             padding = unit(c(6, 4), "mm")
                           ))
grid.draw(business_grob)

# Add recommendations
grid.text("Recommendations:", 
          x = 0.1, y = 0.25, 
          just = "left",
          gp = gpar(fontsize = 14, fontface = "bold"))

recommendations <- c(
  "• Model shows strong fraud detection capabilities",
  "• Consider the trade-off between detection rate and false alarms",
  "• Monitor false positive rate for customer experience",
  "• Regular model retraining recommended for evolving patterns",
  "• Consider ensemble methods if higher precision required"
)

for (i in 1:length(recommendations)) {
  grid.text(recommendations[i], 
            x = 0.1, y = 0.20 - (i * 0.03), 
            just = "left",
            gp = gpar(fontsize = 11))
}

dev.off()

cat("📊 Comprehensive final report saved:", final_test_results_path, "\n")

# Save individual plots
ggsave(final_test_roc_path, roc_plot, width = 8, height = 6)
ggsave(final_test_confusion_path, cm_plot, width = 8, height = 6)

cat("📈 Individual plots saved:\n")
cat("   -", final_test_roc_path, "\n")
cat("   -", final_test_confusion_path, "\n")

# ============================================
# Final Summary and Conclusions
# ============================================

cat("\n")
cat(rep("=", 60), "\n", sep = "")
cat("🎉 FINAL MODEL EVALUATION COMPLETE\n")
cat(rep("=", 60), "\n\n", sep = "")

cat("🏆 MODEL PERFORMANCE SUMMARY:\n")
cat(sprintf("• AUC: %.4f (95%% CI: %.4f - %.4f)\n", test_auc, test_auc_ci[1], test_auc_ci[3]))
cat(sprintf("• Accuracy: %.2f%%\n", accuracy * 100))
cat(sprintf("• Recall (Sensitivity): %.2f%%\n", sensitivity * 100))
cat(sprintf("• Precision: %.2f%%\n", precision * 100))
cat(sprintf("• F1-Score: %.2f%%\n", f1 * 100))
cat(sprintf("• Specificity: %.2f%%\n", specificity * 100))
cat(sprintf("• Balanced Accuracy: %.2f%%\n", balanced_accuracy * 100))

cat("\n📊 BUSINESS IMPACT:\n")
cat(sprintf("• Fraud Detection Rate: %.2f%%\n", detection_rate * 100))
cat(sprintf("• False Alarm Rate: %.2f%%\n", false_alarm_rate * 100))
cat(sprintf("• Fraud Cases Caught: %d / %d\n", true_positives, true_positives + false_negatives))
cat(sprintf("• False Alarms: %d\n", false_positives))

cat("\n✅ CONCLUSION:\n")
if (test_auc > 0.9) {
  cat("• 🎯 EXCELLENT: Model demonstrates outstanding fraud detection capability\n")
} else if (test_auc > 0.8) {
  cat("• 👍 VERY GOOD: Model shows strong fraud detection performance\n")
} else if (test_auc > 0.7) {
  cat("• 🔶 GOOD: Model provides acceptable fraud detection\n")
} else {
  cat("• ⚠️  FAIR: Model may need improvement for production use\n")
}

cat("\n📁 GENERATED FILES:\n")
cat("•", final_test_results_path, "- Comprehensive evaluation report\n")
cat("•", final_test_roc_path, "- ROC curve visualization\n")
cat("•", final_test_confusion_path, "- Confusion matrix\n")
cat("•", final_test_summary_path, "- Performance metrics summary\n")

cat("\n🎯 NEXT STEPS:\n")
cat("• Deploy the Decision Tree model for real-time fraud detection\n")
cat("• Monitor performance metrics in production environment\n")
cat("• Schedule regular model retraining with new data\n")
cat("• Consider A/B testing with other models if needed\n")

cat("\n")
cat(rep("=", 60), "\n", sep = "")
cat("✅ Final evaluation completed successfully!\n")
cat(rep("=", 60), "\n", sep = "")

cat("\n 👍 👍 👍 👍 6104 assignment is completed THXXXXXXXXXXXX👍 👍 👍 👍  \n")

