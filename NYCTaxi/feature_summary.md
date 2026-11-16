# NYC taxi case study: features and summary statistics

Statistics are computed on the two million trips (December 2013 to November 2014, excluding the week of
November 17, 2014) used to fit the simulator (`run.py prepare`). Binary features show the share of 1 and 0.

| # | Variable | Description | Summary statistics |
|---|---|---|---|
|   | fare amount (USD) | Recorded fare of the trip. | range 2.50 to 70.00, median 9.50, mean 11.67, std 7.71 |
| 1 | passenger count | Number of passengers. | mean 1.68, std 1.35 |
| 2 | distance (mi) | Trip distance. | range 0.1 to 29.1, mean 2.66, std 2.64 |
| 3 | pickup in manhattan | 1 if pickup is in Manhattan. | 1 (88.43%), 0 (11.57%) |
| 4 | dropoff in manhattan | 1 if dropoff is in Manhattan. | 1 (85.26%), 0 (14.74%) |
| 5 | pickup near airport | 1 if pickup is at a major airport. | 1 (2.68%), 0 (97.32%) |
| 6 | dropoff near airport | 1 if dropoff is at a major airport. | 1 (1.46%), 0 (98.54%) |
| 7 | traffic congestion | 1 on weekdays from 8 AM to 7 PM. | 1 (38.11%), 0 (61.89%) |
| 8 | pickup in business district | 1 if pickup is in a business district. | 1 (13.57%), 0 (86.43%) |
| 9 | dropoff in business district | 1 if dropoff is in a business district. | 1 (13.45%), 0 (86.55%) |
| 10 | is morning commute | 1 for trips into a business district from elsewhere, excluding airports, 7 to 9 AM. | 1 (1.61%), 0 (98.39%) |
| 11 | is evening commute | 1 for trips from a business district to elsewhere, excluding airports, 5 to 7 PM. | 1 (1.33%), 0 (98.67%) |
| 12 | is biz to airport | 1 for trips from a business district to an airport. | 1 (0.30%), 0 (99.70%) |
| 13 | is airport to biz | 1 for trips from an airport to a business district. | 1 (0.38%), 0 (99.62%) |
| 14 | had rain | 1 if it rained on the day of the trip. | 1 (40.42%), 0 (59.58%) |
| 15 | had snow fall | 1 if it snowed on the day of the trip. | 1 (9.96%), 0 (90.04%) |
| 16 | is extreme temp | 1 if the daily high exceeded 90 °F or the low fell below 32 °F. | 1 (31.69%), 0 (68.31%) |
| 17 | is long daylight | 1 if daylight lasted at least 12 hours. | 1 (54.05%), 0 (45.95%) |
| 18 | is weekend | 1 on Saturdays and Sundays. | 1 (29.49%), 0 (70.51%) |
| 19 | is night | 1 from 10 PM to 5 AM. | 1 (22.92%), 0 (77.08%) |
| 20 | is cross borough | 1 if pickup and dropoff are in different boroughs. | 1 (11.38%), 0 (88.62%) |
