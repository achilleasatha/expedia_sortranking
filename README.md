### Expedia Hotel Ranking - https://www.kaggle.com/c/sortranking/overview
##### Problem statement
Each record of the dataset is an item in the search results and contains both search-level features (the input of the user, his country/city etc) and item-level features (information about the hotel). For each search in this dataset we observe which item was chosen at the end of the session.

Given these features, try and learn a model for hotel relevance.

##### Evaluation
Evaluation Metric is NDCG@50. Relevance is given by propbookingbool
(https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG)

##### Data fields
srch_id: unique identifier for a search <br/>
visitor_id: visitor id used to submit the search <br/>
visitor_visit_nbr: count of visits for the user who has submitted the search <br/>
visitor_loc_country: country of the user who has submitted the search <br/>
visitor_loc_region: region of the user who has submitted the search <br/>
visitor_loc_city: city of the user who has submitted the search <br/>
visitor_wr_member: indicates if user is part of the hcom loyalty program<br/>
posa_continent: point of sale continent for a partiular search<br/>
posa_country: point of sale country for a particular search<br/>
srch_hcom_destination_id: destination id for a particular search <br/>
srch_dest_longitude: destination center longitude for a particular search<br/>
srch_dest_latitude: destination center latitude for a particular search<br/>
srch_ci: date of check-in of a search<br/>
srch_co: date of cehck-in of a search<br/>
srch_ci_day: day of check-in of a search<br/>
srch_co_day: day of check-out of a search<br/>
srch_los: length of stay of a search (check-in - check-out date)<br/>
srch_bw: booking window of a search (search-date - check-in date)<br/>
srch_adults_cnt: number of adults for a search <br/>
srch_children_cnt: number of children for a search <br/>
srch_rm_cnt: number of requested rooms on the search <br/>
mobile_bool: indicates if the search was submitted on a mobile browser<br/>
mobile_app: indicates if the search was submitted on a mobile app<br/>
device: identifies the device type type used for this search<br/>
currency: currency relevant for the point of sale the search was submitted form<br/>
position: the rank this property has been impressed on in a particular search<br/>
travelad_bool: indicates if a porperty in a search was a paid advert<br/>
dotd_bool: indicates if a property in a search was advertised as a special "daily deal"<br/>
price_without_discount_local: discounted property price in local (posa specific) currency<br/>
price_without_discount_usd: discounted property price in USD<br/>
price_with_discount_local: non-discounted property price in local currency<br/>
price_with_discount_usd: non-discounted property price in USD<br/>
imp_drr: name of the pricing/discount rule relevant for this property<br/>
click_bool: indicates if a property has been clicked in a particular search<br/>
booking_bool: indicates if property has been booken in a given search<br/>
prop_key: unique property identifier<br/>
prop_brand_bool: indicates if a property has a popular brand<br/>
prop_starrating: conventional hotel star rating (1-5 Stars)<br/>
prop_super_region: the super region this property belongs to (EMEA, APAC, NA, LATAM)<br/>
prop_continent: continent this property is located on (EU, NA, ...)<br/>
prop_country: country this property is located in<br/>
prop_market_id: higher granularity geo classification<br/>
prop_submarket_id: highest granularity geo classification<br/>
prop_room_capacity: max number of rooms this property has available<br/>
prop_review_score: average guest review score for this property<br/>
prop_review_count: count of customer reviews for this property<br/>
prop_hostel_bool: indicates if a property is a hostel<br/>
local_date: date for a particular search<br/>
