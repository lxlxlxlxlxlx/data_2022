from easydict import EasyDict as edict

VIDEO_GAMES = 'Video_Games'
CELLPHONES = 'Cell_Phones_and_Accessories'
MOVIES = 'Movies_and_TV'
CDS = 'CDs_and_Vinyl'
KINDLE = 'Kindle_Store'
GROCERY = 'Grocery_and_Gourmet_Food'
AUTO = 'Automotive'
TOYS = 'Toys_and_Games'

params = {
    VIDEO_GAMES: {
        'min_purchases': 6,
        'min_purchased_by': 1,
        'num_raw_reviews': 50000,  # for sentries input
        'min_features': 8,
        'min_also_view_buy': 11,
    },
    CELLPHONES: {
        'min_purchases': 7,  # min number of purchase edges for each user.
        'min_purchased_by': 5,  # min number of purchased_by edges for each product.
        'num_raw_reviews': 220000,  # number of reviews for sentries input.
        'min_features': 10,
        'min_also_view_buy': 2,  # min number of also_view/also_buy edges for each related product.
    },
    GROCERY: {
        'min_purchases': 7,  # min number of purchase edges for each user.
        'min_purchased_by': 5,  # min number of purchased_by edges for each product.
        'num_raw_reviews': 300000,
        'min_features': 2,  # min frequency of has_feature edges for each feature.
        'min_also_view_buy': 11,  # min number of also_view/also_buy edges for each related product.
    },
    AUTO: {
        'min_purchases': 7,  # min number of purchase edges for each user.
        'min_purchased_by': 5,  # min number of purchased_by edges for each product.
        'num_raw_reviews': 350000,
        'min_features': 10,
        'min_also_view_buy': 6,  # min number of also_view/also_buy edges for each related product.
    },
    TOYS: {
        'min_purchases': 7,  # min number of purchase edges for each user.
        'min_purchased_by': 5,  # min number of purchased_by edges for each product.
        'num_raw_reviews': 220000,
        'min_features': 10,
        'min_also_view_buy': 11,  # min number of also_view/also_buy edges for each related product.
    },
}

args = edict()


# Dataset name.
args.dataset = CELLPHONES
# args.dataset = AUTO
#args.dataset = GROCERY
# args.dataset = TOYS

args.update(params[args.dataset])

# Raw data directory.
args.data_dir = './data-amazon-2018'

# Raw data files.
args.raw_review_file = '{}/{}_5.json.gz'.format(args.data_dir, args.dataset)
args.raw_meta_file = '{}/meta_{}.json.gz'.format(args.data_dir, args.dataset)

# Generated data directory.
# args.tmp_dir = './data-amazon-2018-tmp/{}'.format(args.dataset)
# args.tmp_dir = './{}/minimal'.format(args.dataset)
args.tmp_dir = './{}'.format(args.dataset)
# Cleaned data files.
args.clean_review_file = '{}/reviews_{}_5.parquet.gzip'.format(args.tmp_dir, args.dataset)
args.clean_meta_file = '{}/meta_{}.parquet.gzip'.format(args.tmp_dir, args.dataset)

# Sentries tool related files.
args.sentiment_raw_file = '{}/sentiment_tool_{}_5.raw'.format(args.tmp_dir, args.dataset)
args.sentiment_product_file = '{}/sentiment_tool_{}_5.product'.format(args.tmp_dir, args.dataset)
args.pos_profile_file = '{}/sentiment_tool_{}_5.pos.profile'.format(args.tmp_dir, args.dataset)
args.neg_profile_file = '{}/sentiment_tool_{}_5.neg.profile'.format(args.tmp_dir, args.dataset)
args.u_i_dict_file = '{}/sentiment_tool_{}_5.pos.profile.u_i_dict'.format(args.tmp_dir, args.dataset)
args.feature_dict_file = '{}/sentiment_tool_{}_5.pos.profile.feature_dict'.format(args.tmp_dir, args.dataset)
args.reference_file = '{}/reference_{}.pkl'.format(args.tmp_dir, args.dataset)
args.dialog_file = '{}/dialog_{}.txt'.format(args.tmp_dir, args.dataset)

# Product related files (generated from metadata).
args.product_desc_file = '{}/product_desc_{}.pkl'.format(args.tmp_dir, args.dataset)
args.product_desc_tfidf_file = '{}/product_desc_tfidf_{}.pkl'.format(args.tmp_dir, args.dataset)
args.product_price_file = '{}/product_price_{}.pkl'.format(args.tmp_dir, args.dataset)
args.product_brand_file = '{}/product_brand_{}.pkl'.format(args.tmp_dir, args.dataset)
args.product_category_file = '{}/product_category_{}.pkl'.format(args.tmp_dir, args.dataset)
args.product_also_view_file = '{}/product_also_view_{}.pkl'.format(args.tmp_dir, args.dataset)
args.product_also_buy_file = '{}/product_also_buy_{}.pkl'.format(args.tmp_dir, args.dataset)
args.product_feature_file = '{}/product_feature_{}.pkl'.format(args.tmp_dir, args.dataset)
args.product_style_file = '{}/product_style_{}.pkl'.format(args.tmp_dir, args.dataset)

# args.product_desc_raw_file = '{}/product_desc_{}.raw'.format(args.tmp_dir, args.dataset)
# args.product_desc_product_file = '{}/product_desc_{}.product'.format(args.tmp_dir, args.dataset)

# KG constants
args.entity_user = 'user::'
args.entity_product = 'product::'
args.entity_aspect_val = 'aspect_value::'
args.entity_feature = 'feature::'
args.entity_price = 'price::'
args.entity_brand = 'brand::'
args.entity_category = 'category::'
args.entity_related_product = 'related_product::'
args.entity_style = 'style::'
args.rel_purchase = 'purchase'
args.rel_review = 'review'
args.rel_reviewed_by = 'reviewed_by'
args.rel_has_feature = 'has_feature'
args.rel_has_price = 'has_price'
args.rel_also_view = 'also_view'
args.rel_also_buy = 'also_buy'
args.rel_has_brand = 'has_brand'
args.rel_has_category = 'has_category'
args.rel_has_style = 'has_style::'  # This is prefix for relation product -> style.
args.rel_like_style = 'like_style::'  # This is prefix for relation user -> style.

# KG related files
args.train_set_file = '{}/train_set_{}.pkl'.format(args.tmp_dir, args.dataset)
args.val_set_file = '{}/validate_set_{}.pkl'.format(args.tmp_dir, args.dataset)
args.test_set_file = '{}/test_set_{}.pkl'.format(args.tmp_dir, args.dataset)
args.kg_users_file = '{}/kg_users_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_items_file = '{}/kg_items_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_entities_file = '{}/kg_entities_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_relations_file = '{}/kg_relations_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_train_triples_file = '{}/kg_train_triples_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_val_triples_file = '{}/kg_val_triples_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_test_triples_file = '{}/kg_test_triples_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_other_triples_file = '{}/kg_other_triples_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_train_candidates_file = '{}/kg_train_candidates_{}.txt.gz'.format(args.tmp_dir, args.dataset)  # this is useless
args.kg_val_candidates_file = '{}/kg_val_candidates_{}.txt.gz'.format(args.tmp_dir, args.dataset)
args.kg_test_candidates_file = '{}/kg_test_candidates_{}.txt.gz'.format(args.tmp_dir, args.dataset)
#args.kg_test_candidates_file = '{}/rec_test_candidate100.npz'.format(args.tmp_dir)
args.kg_train_gt_file = '{}/kg_train_gt_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_val_gt_file = '{}/kg_val_gt_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_test_gt_file = '{}/kg_test_gt_{}.txt'.format(args.tmp_dir, args.dataset)
