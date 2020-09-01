# Expand training rating

# 1. Shuffle

Movie-lens data are not fit for time-series models.

Total number of lines: 1000209

Split into train, val, test as 6:2:2.

> head -600209 shuf_ratings.dat > sorted_train.dat
> head -800209 shuf_ratings.dat | tail -200000 > sorted_val.dat
> tail -200000 shuf_ratings.dat > sorted_test.dat
> head -800209 shuf_ratings.dat > sorted_train_val.dat

# 2. Use user and item in train_val to do one-hot encoder.

> To produce  
> user_id, movie_id, rating, timestamp
> + user features
> gender_id, age_period_id, occupation_id, zip_code_id
> + movie features
> genres_id
> TODO: title  
   
 + Get user_dict and item_dict of train_val_ratings.
 
 > python get_train_meta_id.py
 
> user_dict: {user_str: user_id}
>
> item_dict: {item_str: item_id}
 
> #user_dict = 4796
>
> #movie_dict = 3685
   
 + Get age_period_dict, occupation_dict, zip_code_dict from users.dat
 
 > gender_dict: {'M': 1, 'F': 0}
 
 + TODO, Get genres_dict from movies.dat
 
 genres are multi-labels.
 
 # 3. Concat features for train, val and test data.
 
 > python concat_feature.py
 
 > Note: Index is accumulated, (movie_index = movie_id + num_user).
 
 + Load users.dat and movies.dat, and all of the mapping_dict with gender_dict.
  
 + For each sample `user::movie::rating::timestamp`, get the id and feature_id.
 
 