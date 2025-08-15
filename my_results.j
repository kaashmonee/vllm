{
  "experiment_metadata": {
    "timestamp": "2025-08-15T00:59:56.739182",
    "model": "./examples/offline_inference/basic/model_cache/llama-3.1-8b",
    "dataset": "ccdv/patent-classification",
    "optimization_version": "enhanced_v2",
    "total_classes": 9,
    "class_names": {
      "0": "Human Necessities",
      "1": "Performing Operations; Transporting",
      "2": "Chemistry; Metallurgy",
      "3": "Textiles; Paper",
      "4": "Fixed Constructions",
      "5": "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
      "6": "Physics",
      "7": "Electricity",
      "8": "General tagging of new or cross-sectional technology"
    },
    "enhanced_descriptions": {
      "0": {
        "name": "Human Necessities",
        "description": "Food, beverages, tobacco, clothing, footwear, headgear, shelter, health, life-saving, amusement, sports",
        "keywords": [
          "food",
          "beverage",
          "clothing",
          "medical",
          "health",
          "drug",
          "pharmaceutical",
          "game",
          "sport",
          "entertainment",
          "household",
          "furniture",
          "kitchen",
          "cosmetic",
          "hygiene"
        ],
        "distinctive": [
          "patient",
          "treatment",
          "medicine",
          "pharmaceutical",
          "food",
          "cooking",
          "clothing",
          "game",
          "toy"
        ]
      },
      "1": {
        "name": "Performing Operations; Transporting",
        "description": "Manufacturing processes, machines, tools, vehicles, transportation systems, logistics",
        "keywords": [
          "machine",
          "manufacturing",
          "processing",
          "tool",
          "vehicle",
          "transport",
          "engine",
          "motor",
          "pump",
          "conveyor",
          "assembly",
          "production"
        ],
        "distinctive": [
          "manufacturing",
          "machining",
          "conveyor",
          "assembly",
          "vehicle",
          "transport",
          "logistics",
          "pump",
          "compressor"
        ]
      },
      "2": {
        "name": "Chemistry; Metallurgy",
        "description": "Chemical compounds, reactions, materials, alloys, metallurgical processes",
        "keywords": [
          "chemical",
          "compound",
          "reaction",
          "catalyst",
          "polymer",
          "metal",
          "alloy",
          "synthesis",
          "composition",
          "material",
          "substance",
          "molecular"
        ],
        "distinctive": [
          "chemical",
          "compound",
          "polymer",
          "catalyst",
          "synthesis",
          "molecular",
          "alloy",
          "metallurgy",
          "composition"
        ]
      },
      "3": {
        "name": "Textiles; Paper",
        "description": "Fabrics, fibers, yarns, weaving, paper manufacturing, pulp processing",
        "keywords": [
          "fabric",
          "textile",
          "fiber",
          "yarn",
          "weaving",
          "paper",
          "pulp",
          "thread",
          "cloth",
          "cotton",
          "wool",
          "silk"
        ],
        "distinctive": [
          "textile",
          "fabric",
          "fiber",
          "yarn",
          "weaving",
          "paper",
          "pulp",
          "cloth",
          "thread",
          "spinning"
        ]
      },
      "4": {
        "name": "Fixed Constructions",
        "description": "Buildings, structures, foundations, roofs, walls, bridges, roads, construction methods",
        "keywords": [
          "building",
          "construction",
          "structure",
          "foundation",
          "roof",
          "wall",
          "bridge",
          "road",
          "concrete",
          "steel",
          "beam",
          "foundation"
        ],
        "distinctive": [
          "building",
          "construction",
          "structure",
          "foundation",
          "concrete",
          "beam",
          "roof",
          "wall",
          "bridge",
          "road"
        ]
      },
      "5": {
        "name": "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
        "description": "Mechanical systems, lighting, heating, cooling, ventilation, weapons, explosive devices",
        "keywords": [
          "mechanical",
          "gear",
          "bearing",
          "heating",
          "cooling",
          "ventilation",
          "lighting",
          "lamp",
          "hvac",
          "turbine",
          "valve",
          "piston"
        ],
        "distinctive": [
          "mechanical",
          "gear",
          "bearing",
          "valve",
          "piston",
          "heating",
          "cooling",
          "ventilation",
          "lighting",
          "turbine",
          "hvac"
        ]
      },
      "6": {
        "name": "Physics",
        "description": "Scientific instruments, optics, photography, cinematography, measuring, testing, navigation",
        "keywords": [
          "instrument",
          "optical",
          "lens",
          "camera",
          "measurement",
          "sensor",
          "detector",
          "laser",
          "microscope",
          "telescope",
          "photography"
        ],
        "distinctive": [
          "optical",
          "lens",
          "camera",
          "measurement",
          "sensor",
          "detector",
          "laser",
          "microscope",
          "instrument",
          "photography"
        ]
      },
      "7": {
        "name": "Electricity",
        "description": "Electrical circuits, electronics, power generation, transmission, communication, computing",
        "keywords": [
          "electrical",
          "electronic",
          "circuit",
          "power",
          "current",
          "voltage",
          "battery",
          "generator",
          "transformer",
          "semiconductor",
          "computer",
          "communication"
        ],
        "distinctive": [
          "electrical",
          "electronic",
          "circuit",
          "power",
          "voltage",
          "battery",
          "semiconductor",
          "computer",
          "processor",
          "communication"
        ]
      },
      "8": {
        "name": "General tagging of new or cross-sectional technology",
        "description": "Emerging technologies, nanotechnology, biotechnology, cross-disciplinary innovations",
        "keywords": [
          "nanotechnology",
          "biotechnology",
          "emerging",
          "innovative",
          "cross-sectional",
          "interdisciplinary",
          "novel",
          "advanced"
        ],
        "distinctive": [
          "nanotechnology",
          "biotechnology",
          "emerging",
          "novel",
          "innovative",
          "cross-sectional",
          "interdisciplinary",
          "advanced"
        ]
      }
    },
    "dataset_info": {
      "total_samples": 1493,
      "class_distribution": {
        "0": 160,
        "1": 205,
        "2": 143,
        "3": 102,
        "4": 109,
        "5": 124,
        "6": 256,
        "7": 204,
        "8": 190
      },
      "sampling_strategy": "advanced_stratified"
    },
    "experiment_config": {
      "few_shot_configs_tested": [
        1,
        2,
        3,
        5,
        7
      ],
      "enhancement_combinations": [
        "baseline",
        "chain_of_thought",
        "multiple_sampling",
        "both"
      ],
      "confidence_threshold": 0.6,
      "samples_per_prediction": 3,
      "random_seed": 42
    }
  },
  "results_summary": {
    "best_config": {
      "few_shot_count": 1,
      "accuracy": 0.4032150033489618,
      "confidence": 0.9841482473766466,
      "enhancements": {
        "chain_of_thought": true,
        "multiple_sampling": true,
        "confidence_threshold": 0.6
      }
    },
    "worst_config": {
      "few_shot_count": 1,
      "accuracy": 0.3543201607501674
    },
    "overall_statistics": {
      "mean_accuracy": 0.38008414617288133,
      "accuracy_std": 0.015483048050199169,
      "accuracy_range": [
        0.3543201607501674,
        0.4032150033489618
      ],
      "improvement_over_baseline": 0.0488948425987944
    }
  },
  "detailed_results": [
    {
      "few_shot_count": 1,
      "total_samples": 1493,
      "valid_predictions": 1493,
      "invalid_predictions": 0,
      "overall_accuracy": 0.3543201607501674,
      "correct_predictions": 529,
      "average_confidence": 1.0,
      "per_class_metrics": {
        "0": {
          "precision": 0.24703557312252963,
          "recall": 0.78125,
          "f1": 0.3753753753753754,
          "support": 160
        },
        "1": {
          "precision": 0.3033175355450237,
          "recall": 0.3121951219512195,
          "f1": 0.30769230769230765,
          "support": 205
        },
        "2": {
          "precision": 0.5064102564102564,
          "recall": 0.5524475524475524,
          "f1": 0.5284280936454848,
          "support": 143
        },
        "3": {
          "precision": 0.6557377049180327,
          "recall": 0.39215686274509803,
          "f1": 0.4907975460122699,
          "support": 102
        },
        "4": {
          "precision": 0.4025974025974026,
          "recall": 0.28440366972477066,
          "f1": 0.3333333333333333,
          "support": 109
        },
        "5": {
          "precision": 0.3125,
          "recall": 0.28225806451612906,
          "f1": 0.29661016949152547,
          "support": 124
        },
        "6": {
          "precision": 0.4956521739130435,
          "recall": 0.22265625,
          "f1": 0.307277628032345,
          "support": 256
        },
        "7": {
          "precision": 0.4791666666666667,
          "recall": 0.45098039215686275,
          "f1": 0.46464646464646464,
          "support": 204
        },
        "8": {
          "precision": 0.09523809523809523,
          "recall": 0.031578947368421054,
          "f1": 0.047430830039525695,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          125,
          6,
          10,
          1,
          4,
          3,
          9,
          1,
          1
        ],
        [
          72,
          64,
          17,
          8,
          20,
          15,
          2,
          4,
          3
        ],
        [
          44,
          6,
          79,
          2,
          1,
          3,
          3,
          1,
          4
        ],
        [
          34,
          6,
          6,
          40,
          0,
          10,
          0,
          5,
          1
        ],
        [
          47,
          24,
          0,
          0,
          31,
          5,
          0,
          0,
          2
        ],
        [
          28,
          43,
          4,
          2,
          6,
          35,
          1,
          4,
          1
        ],
        [
          68,
          13,
          2,
          4,
          2,
          9,
          57,
          69,
          32
        ],
        [
          36,
          10,
          6,
          0,
          0,
          14,
          33,
          92,
          13
        ],
        [
          52,
          39,
          32,
          4,
          13,
          18,
          10,
          16,
          6
        ]
      ],
      "processing_time_seconds": 40.17457437515259,
      "samples_per_second": 37.16280814970873,
      "enhancements_used": {
        "chain_of_thought": false,
        "multiple_sampling": false,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 1,
      "total_samples": 1493,
      "valid_predictions": 1493,
      "invalid_predictions": 0,
      "overall_accuracy": 0.39986604152712657,
      "correct_predictions": 597,
      "average_confidence": 1.0,
      "per_class_metrics": {
        "0": {
          "precision": 0.3706293706293706,
          "recall": 0.6625,
          "f1": 0.4753363228699552,
          "support": 160
        },
        "1": {
          "precision": 0.3558282208588957,
          "recall": 0.28292682926829266,
          "f1": 0.31521739130434784,
          "support": 205
        },
        "2": {
          "precision": 0.5144508670520231,
          "recall": 0.6223776223776224,
          "f1": 0.5632911392405062,
          "support": 143
        },
        "3": {
          "precision": 0.6612903225806451,
          "recall": 0.4019607843137255,
          "f1": 0.5,
          "support": 102
        },
        "4": {
          "precision": 0.4625,
          "recall": 0.3394495412844037,
          "f1": 0.3915343915343915,
          "support": 109
        },
        "5": {
          "precision": 0.33678756476683935,
          "recall": 0.5241935483870968,
          "f1": 0.4100946372239747,
          "support": 124
        },
        "6": {
          "precision": 0.46774193548387094,
          "recall": 0.2265625,
          "f1": 0.30526315789473685,
          "support": 256
        },
        "7": {
          "precision": 0.38920454545454547,
          "recall": 0.6715686274509803,
          "f1": 0.4928057553956834,
          "support": 204
        },
        "8": {
          "precision": 0.1,
          "recall": 0.031578947368421054,
          "f1": 0.048,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          106,
          6,
          12,
          1,
          5,
          9,
          12,
          5,
          4
        ],
        [
          43,
          58,
          20,
          10,
          16,
          32,
          5,
          14,
          7
        ],
        [
          30,
          4,
          89,
          2,
          2,
          2,
          2,
          4,
          8
        ],
        [
          27,
          2,
          6,
          41,
          2,
          14,
          0,
          9,
          1
        ],
        [
          24,
          23,
          2,
          1,
          37,
          14,
          0,
          2,
          6
        ],
        [
          10,
          25,
          3,
          1,
          5,
          65,
          3,
          8,
          4
        ],
        [
          20,
          10,
          3,
          1,
          1,
          19,
          58,
          133,
          11
        ],
        [
          1,
          6,
          5,
          0,
          0,
          10,
          32,
          137,
          13
        ],
        [
          25,
          29,
          33,
          5,
          12,
          28,
          12,
          40,
          6
        ]
      ],
      "processing_time_seconds": 53.15428280830383,
      "samples_per_second": 28.088047117188488,
      "enhancements_used": {
        "chain_of_thought": true,
        "multiple_sampling": false,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 1,
      "total_samples": 1493,
      "valid_predictions": 1492,
      "invalid_predictions": 1,
      "overall_accuracy": 0.3552278820375335,
      "correct_predictions": 530,
      "average_confidence": 0.9832439678284183,
      "per_class_metrics": {
        "0": {
          "precision": 0.250501002004008,
          "recall": 0.78125,
          "f1": 0.3793626707132018,
          "support": 160
        },
        "1": {
          "precision": 0.2966507177033493,
          "recall": 0.3024390243902439,
          "f1": 0.2995169082125604,
          "support": 205
        },
        "2": {
          "precision": 0.512987012987013,
          "recall": 0.5524475524475524,
          "f1": 0.531986531986532,
          "support": 143
        },
        "3": {
          "precision": 0.6507936507936508,
          "recall": 0.4019607843137255,
          "f1": 0.496969696969697,
          "support": 102
        },
        "4": {
          "precision": 0.3780487804878049,
          "recall": 0.28703703703703703,
          "f1": 0.32631578947368417,
          "support": 108
        },
        "5": {
          "precision": 0.30275229357798167,
          "recall": 0.2661290322580645,
          "f1": 0.2832618025751073,
          "support": 124
        },
        "6": {
          "precision": 0.5087719298245614,
          "recall": 0.2265625,
          "f1": 0.31351351351351353,
          "support": 256
        },
        "7": {
          "precision": 0.4797979797979798,
          "recall": 0.46568627450980393,
          "f1": 0.472636815920398,
          "support": 204
        },
        "8": {
          "precision": 0.09375,
          "recall": 0.031578947368421054,
          "f1": 0.04724409448818898,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          125,
          5,
          10,
          1,
          4,
          3,
          9,
          1,
          2
        ],
        [
          72,
          62,
          17,
          8,
          23,
          14,
          3,
          4,
          2
        ],
        [
          46,
          6,
          79,
          2,
          1,
          3,
          1,
          1,
          4
        ],
        [
          35,
          6,
          5,
          41,
          0,
          10,
          0,
          4,
          1
        ],
        [
          45,
          23,
          0,
          0,
          31,
          6,
          0,
          0,
          3
        ],
        [
          29,
          43,
          4,
          2,
          6,
          33,
          2,
          4,
          1
        ],
        [
          66,
          14,
          2,
          4,
          2,
          7,
          58,
          72,
          31
        ],
        [
          32,
          10,
          6,
          1,
          0,
          14,
          32,
          95,
          14
        ],
        [
          49,
          40,
          31,
          4,
          15,
          19,
          9,
          17,
          6
        ]
      ],
      "processing_time_seconds": 53.503018379211426,
      "samples_per_second": 27.90496770515109,
      "enhancements_used": {
        "chain_of_thought": false,
        "multiple_sampling": true,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 1,
      "total_samples": 1493,
      "valid_predictions": 1493,
      "invalid_predictions": 0,
      "overall_accuracy": 0.4032150033489618,
      "correct_predictions": 602,
      "average_confidence": 0.9841482473766466,
      "per_class_metrics": {
        "0": {
          "precision": 0.37809187279151946,
          "recall": 0.66875,
          "f1": 0.48306997742663654,
          "support": 160
        },
        "1": {
          "precision": 0.36024844720496896,
          "recall": 0.28292682926829266,
          "f1": 0.3169398907103825,
          "support": 205
        },
        "2": {
          "precision": 0.5056179775280899,
          "recall": 0.6293706293706294,
          "f1": 0.5607476635514019,
          "support": 143
        },
        "3": {
          "precision": 0.6833333333333333,
          "recall": 0.4019607843137255,
          "f1": 0.5061728395061729,
          "support": 102
        },
        "4": {
          "precision": 0.47560975609756095,
          "recall": 0.3577981651376147,
          "f1": 0.4083769633507853,
          "support": 109
        },
        "5": {
          "precision": 0.32275132275132273,
          "recall": 0.49193548387096775,
          "f1": 0.389776357827476,
          "support": 124
        },
        "6": {
          "precision": 0.5040650406504065,
          "recall": 0.2421875,
          "f1": 0.32717678100263853,
          "support": 256
        },
        "7": {
          "precision": 0.3865546218487395,
          "recall": 0.6764705882352942,
          "f1": 0.4919786096256685,
          "support": 204
        },
        "8": {
          "precision": 0.1,
          "recall": 0.031578947368421054,
          "f1": 0.048,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          107,
          6,
          12,
          1,
          4,
          9,
          12,
          5,
          4
        ],
        [
          43,
          58,
          21,
          8,
          17,
          32,
          4,
          15,
          7
        ],
        [
          29,
          4,
          90,
          2,
          2,
          2,
          2,
          5,
          7
        ],
        [
          27,
          4,
          6,
          41,
          2,
          12,
          0,
          9,
          1
        ],
        [
          23,
          21,
          2,
          2,
          39,
          13,
          0,
          3,
          6
        ],
        [
          10,
          26,
          4,
          1,
          5,
          61,
          3,
          10,
          4
        ],
        [
          19,
          10,
          3,
          1,
          1,
          17,
          62,
          131,
          12
        ],
        [
          0,
          3,
          6,
          0,
          0,
          15,
          29,
          138,
          13
        ],
        [
          25,
          29,
          34,
          4,
          12,
          28,
          11,
          41,
          6
        ]
      ],
      "processing_time_seconds": 66.5063533782959,
      "samples_per_second": 22.448983054410483,
      "enhancements_used": {
        "chain_of_thought": true,
        "multiple_sampling": true,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 2,
      "total_samples": 1493,
      "valid_predictions": 1493,
      "invalid_predictions": 0,
      "overall_accuracy": 0.3643670462156731,
      "correct_predictions": 544,
      "average_confidence": 1.0,
      "per_class_metrics": {
        "0": {
          "precision": 0.2588495575221239,
          "recall": 0.73125,
          "f1": 0.38235294117647056,
          "support": 160
        },
        "1": {
          "precision": 0.29891304347826086,
          "recall": 0.2682926829268293,
          "f1": 0.28277634961439585,
          "support": 205
        },
        "2": {
          "precision": 0.45454545454545453,
          "recall": 0.6643356643356644,
          "f1": 0.5397727272727273,
          "support": 143
        },
        "3": {
          "precision": 0.5857142857142857,
          "recall": 0.4019607843137255,
          "f1": 0.47674418604651164,
          "support": 102
        },
        "4": {
          "precision": 0.3918918918918919,
          "recall": 0.26605504587155965,
          "f1": 0.31693989071038253,
          "support": 109
        },
        "5": {
          "precision": 0.410958904109589,
          "recall": 0.24193548387096775,
          "f1": 0.3045685279187817,
          "support": 124
        },
        "6": {
          "precision": 0.4852941176470588,
          "recall": 0.2578125,
          "f1": 0.336734693877551,
          "support": 256
        },
        "7": {
          "precision": 0.49261083743842365,
          "recall": 0.49019607843137253,
          "f1": 0.4914004914004914,
          "support": 204
        },
        "8": {
          "precision": 0.11956521739130435,
          "recall": 0.05789473684210526,
          "f1": 0.07801418439716314,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          117,
          6,
          13,
          1,
          8,
          1,
          11,
          2,
          1
        ],
        [
          65,
          55,
          26,
          18,
          17,
          10,
          5,
          6,
          3
        ],
        [
          33,
          3,
          95,
          1,
          1,
          0,
          2,
          0,
          8
        ],
        [
          29,
          8,
          13,
          41,
          0,
          4,
          2,
          4,
          1
        ],
        [
          45,
          24,
          2,
          0,
          29,
          8,
          0,
          0,
          1
        ],
        [
          29,
          38,
          7,
          2,
          6,
          30,
          5,
          7,
          0
        ],
        [
          52,
          11,
          7,
          3,
          1,
          4,
          66,
          67,
          45
        ],
        [
          28,
          4,
          10,
          0,
          1,
          4,
          35,
          100,
          22
        ],
        [
          54,
          35,
          36,
          4,
          11,
          12,
          10,
          17,
          11
        ]
      ],
      "processing_time_seconds": 42.90992093086243,
      "samples_per_second": 34.79381848327244,
      "enhancements_used": {
        "chain_of_thought": false,
        "multiple_sampling": false,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 2,
      "total_samples": 1493,
      "valid_predictions": 1493,
      "invalid_predictions": 0,
      "overall_accuracy": 0.39517749497655724,
      "correct_predictions": 590,
      "average_confidence": 1.0,
      "per_class_metrics": {
        "0": {
          "precision": 0.49707602339181284,
          "recall": 0.53125,
          "f1": 0.513595166163142,
          "support": 160
        },
        "1": {
          "precision": 0.4297520661157025,
          "recall": 0.25365853658536586,
          "f1": 0.3190184049079755,
          "support": 205
        },
        "2": {
          "precision": 0.452991452991453,
          "recall": 0.7412587412587412,
          "f1": 0.5623342175066314,
          "support": 143
        },
        "3": {
          "precision": 0.6338028169014085,
          "recall": 0.4411764705882353,
          "f1": 0.5202312138728324,
          "support": 102
        },
        "4": {
          "precision": 0.4305555555555556,
          "recall": 0.28440366972477066,
          "f1": 0.3425414364640884,
          "support": 109
        },
        "5": {
          "precision": 0.26693227091633465,
          "recall": 0.5403225806451613,
          "f1": 0.35733333333333334,
          "support": 124
        },
        "6": {
          "precision": 0.5,
          "recall": 0.18359375,
          "f1": 0.26857142857142857,
          "support": 256
        },
        "7": {
          "precision": 0.39325842696629215,
          "recall": 0.6862745098039216,
          "f1": 0.5,
          "support": 204
        },
        "8": {
          "precision": 0.13821138211382114,
          "recall": 0.08947368421052632,
          "f1": 0.10862619808306709,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          85,
          5,
          15,
          1,
          9,
          21,
          10,
          7,
          7
        ],
        [
          18,
          52,
          31,
          13,
          16,
          44,
          3,
          17,
          11
        ],
        [
          17,
          2,
          106,
          2,
          1,
          1,
          2,
          1,
          11
        ],
        [
          14,
          4,
          17,
          45,
          1,
          11,
          0,
          9,
          1
        ],
        [
          8,
          14,
          5,
          2,
          31,
          36,
          0,
          2,
          11
        ],
        [
          3,
          14,
          9,
          1,
          3,
          67,
          2,
          18,
          7
        ],
        [
          11,
          9,
          4,
          2,
          2,
          16,
          47,
          129,
          36
        ],
        [
          0,
          2,
          7,
          0,
          0,
          11,
          22,
          140,
          22
        ],
        [
          15,
          19,
          40,
          5,
          9,
          44,
          8,
          33,
          17
        ]
      ],
      "processing_time_seconds": 66.13957953453064,
      "samples_per_second": 22.573472805652226,
      "enhancements_used": {
        "chain_of_thought": true,
        "multiple_sampling": false,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 2,
      "total_samples": 1493,
      "valid_predictions": 1489,
      "invalid_predictions": 4,
      "overall_accuracy": 0.36601746138347885,
      "correct_predictions": 545,
      "average_confidence": 0.9789567942690843,
      "per_class_metrics": {
        "0": {
          "precision": 0.2621145374449339,
          "recall": 0.7484276729559748,
          "f1": 0.38825448613376834,
          "support": 159
        },
        "1": {
          "precision": 0.2962962962962963,
          "recall": 0.27450980392156865,
          "f1": 0.28498727735368956,
          "support": 204
        },
        "2": {
          "precision": 0.45454545454545453,
          "recall": 0.6643356643356644,
          "f1": 0.5397727272727273,
          "support": 143
        },
        "3": {
          "precision": 0.6,
          "recall": 0.4117647058823529,
          "f1": 0.4883720930232558,
          "support": 102
        },
        "4": {
          "precision": 0.4117647058823529,
          "recall": 0.25925925925925924,
          "f1": 0.3181818181818182,
          "support": 108
        },
        "5": {
          "precision": 0.42424242424242425,
          "recall": 0.22580645161290322,
          "f1": 0.2947368421052632,
          "support": 124
        },
        "6": {
          "precision": 0.4962962962962963,
          "recall": 0.26171875,
          "f1": 0.3427109974424553,
          "support": 256
        },
        "7": {
          "precision": 0.4854368932038835,
          "recall": 0.49019607843137253,
          "f1": 0.4878048780487805,
          "support": 204
        },
        "8": {
          "precision": 0.10869565217391304,
          "recall": 0.05291005291005291,
          "f1": 0.0711743772241993,
          "support": 189
        }
      },
      "confusion_matrix": [
        [
          119,
          7,
          14,
          1,
          4,
          1,
          9,
          2,
          2
        ],
        [
          66,
          56,
          26,
          17,
          16,
          8,
          5,
          7,
          3
        ],
        [
          34,
          3,
          95,
          1,
          1,
          0,
          2,
          0,
          7
        ],
        [
          30,
          6,
          13,
          42,
          0,
          3,
          3,
          4,
          1
        ],
        [
          45,
          25,
          2,
          0,
          28,
          7,
          0,
          0,
          1
        ],
        [
          28,
          39,
          8,
          2,
          6,
          28,
          5,
          8,
          0
        ],
        [
          54,
          11,
          6,
          3,
          1,
          4,
          67,
          68,
          42
        ],
        [
          27,
          4,
          9,
          0,
          0,
          4,
          34,
          100,
          26
        ],
        [
          51,
          38,
          36,
          4,
          12,
          11,
          10,
          17,
          10
        ]
      ],
      "processing_time_seconds": 57.184314489364624,
      "samples_per_second": 26.108558148015824,
      "enhancements_used": {
        "chain_of_thought": false,
        "multiple_sampling": true,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 2,
      "total_samples": 1493,
      "valid_predictions": 1490,
      "invalid_predictions": 3,
      "overall_accuracy": 0.3859060402684564,
      "correct_predictions": 575,
      "average_confidence": 0.9738255033557047,
      "per_class_metrics": {
        "0": {
          "precision": 0.4941860465116279,
          "recall": 0.53125,
          "f1": 0.5120481927710843,
          "support": 160
        },
        "1": {
          "precision": 0.40869565217391307,
          "recall": 0.23039215686274508,
          "f1": 0.2946708463949843,
          "support": 204
        },
        "2": {
          "precision": 0.44396551724137934,
          "recall": 0.7202797202797203,
          "f1": 0.5493333333333333,
          "support": 143
        },
        "3": {
          "precision": 0.625,
          "recall": 0.4411764705882353,
          "f1": 0.5172413793103449,
          "support": 102
        },
        "4": {
          "precision": 0.43661971830985913,
          "recall": 0.28440366972477066,
          "f1": 0.3444444444444445,
          "support": 109
        },
        "5": {
          "precision": 0.25190839694656486,
          "recall": 0.532258064516129,
          "f1": 0.34196891191709844,
          "support": 124
        },
        "6": {
          "precision": 0.4777777777777778,
          "recall": 0.16862745098039217,
          "f1": 0.24927536231884062,
          "support": 255
        },
        "7": {
          "precision": 0.39215686274509803,
          "recall": 0.6862745098039216,
          "f1": 0.4991087344028521,
          "support": 204
        },
        "8": {
          "precision": 0.12605042016806722,
          "recall": 0.07936507936507936,
          "f1": 0.0974025974025974,
          "support": 189
        }
      },
      "confusion_matrix": [
        [
          85,
          6,
          16,
          1,
          8,
          21,
          9,
          7,
          7
        ],
        [
          19,
          47,
          30,
          13,
          16,
          49,
          3,
          16,
          11
        ],
        [
          18,
          2,
          103,
          2,
          1,
          1,
          2,
          3,
          11
        ],
        [
          14,
          4,
          17,
          45,
          0,
          12,
          0,
          9,
          1
        ],
        [
          8,
          11,
          5,
          3,
          31,
          39,
          0,
          0,
          12
        ],
        [
          3,
          16,
          9,
          1,
          3,
          66,
          2,
          17,
          7
        ],
        [
          10,
          8,
          6,
          2,
          2,
          18,
          43,
          132,
          34
        ],
        [
          0,
          2,
          7,
          0,
          0,
          11,
          23,
          140,
          21
        ],
        [
          15,
          19,
          39,
          5,
          10,
          45,
          8,
          33,
          15
        ]
      ],
      "processing_time_seconds": 80.49716377258301,
      "samples_per_second": 18.5472373190434,
      "enhancements_used": {
        "chain_of_thought": true,
        "multiple_sampling": true,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 3,
      "total_samples": 1493,
      "valid_predictions": 1493,
      "invalid_predictions": 0,
      "overall_accuracy": 0.3724045545880777,
      "correct_predictions": 556,
      "average_confidence": 1.0,
      "per_class_metrics": {
        "0": {
          "precision": 0.29048843187660667,
          "recall": 0.70625,
          "f1": 0.4116575591985428,
          "support": 160
        },
        "1": {
          "precision": 0.2962962962962963,
          "recall": 0.23414634146341465,
          "f1": 0.2615803814713897,
          "support": 205
        },
        "2": {
          "precision": 0.4604651162790698,
          "recall": 0.6923076923076923,
          "f1": 0.553072625698324,
          "support": 143
        },
        "3": {
          "precision": 0.7692307692307693,
          "recall": 0.39215686274509803,
          "f1": 0.5194805194805195,
          "support": 102
        },
        "4": {
          "precision": 0.45054945054945056,
          "recall": 0.3761467889908257,
          "f1": 0.41000000000000003,
          "support": 109
        },
        "5": {
          "precision": 0.33653846153846156,
          "recall": 0.28225806451612906,
          "f1": 0.30701754385964913,
          "support": 124
        },
        "6": {
          "precision": 0.5037593984962406,
          "recall": 0.26171875,
          "f1": 0.3444730077120823,
          "support": 256
        },
        "7": {
          "precision": 0.458128078817734,
          "recall": 0.45588235294117646,
          "f1": 0.45700245700245706,
          "support": 204
        },
        "8": {
          "precision": 0.1388888888888889,
          "recall": 0.10526315789473684,
          "f1": 0.11976047904191615,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          113,
          7,
          13,
          1,
          5,
          3,
          11,
          2,
          5
        ],
        [
          63,
          48,
          29,
          7,
          23,
          17,
          3,
          6,
          9
        ],
        [
          28,
          2,
          99,
          1,
          1,
          0,
          2,
          0,
          10
        ],
        [
          32,
          4,
          13,
          40,
          1,
          4,
          0,
          7,
          1
        ],
        [
          30,
          22,
          2,
          0,
          41,
          11,
          0,
          0,
          3
        ],
        [
          23,
          36,
          5,
          1,
          6,
          35,
          7,
          9,
          2
        ],
        [
          37,
          7,
          6,
          2,
          2,
          7,
          67,
          69,
          59
        ],
        [
          16,
          8,
          11,
          0,
          0,
          9,
          32,
          93,
          35
        ],
        [
          47,
          28,
          37,
          0,
          12,
          18,
          11,
          17,
          20
        ]
      ],
      "processing_time_seconds": 43.95943593978882,
      "samples_per_second": 33.96312914580979,
      "enhancements_used": {
        "chain_of_thought": false,
        "multiple_sampling": false,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 3,
      "total_samples": 1493,
      "valid_predictions": 1493,
      "invalid_predictions": 0,
      "overall_accuracy": 0.391158740790355,
      "correct_predictions": 584,
      "average_confidence": 1.0,
      "per_class_metrics": {
        "0": {
          "precision": 0.4943181818181818,
          "recall": 0.54375,
          "f1": 0.5178571428571428,
          "support": 160
        },
        "1": {
          "precision": 0.4180327868852459,
          "recall": 0.24878048780487805,
          "f1": 0.3119266055045872,
          "support": 205
        },
        "2": {
          "precision": 0.43673469387755104,
          "recall": 0.7482517482517482,
          "f1": 0.5515463917525774,
          "support": 143
        },
        "3": {
          "precision": 0.6973684210526315,
          "recall": 0.5196078431372549,
          "f1": 0.5955056179775281,
          "support": 102
        },
        "4": {
          "precision": 0.46774193548387094,
          "recall": 0.26605504587155965,
          "f1": 0.3391812865497076,
          "support": 109
        },
        "5": {
          "precision": 0.2682926829268293,
          "recall": 0.532258064516129,
          "f1": 0.3567567567567567,
          "support": 124
        },
        "6": {
          "precision": 0.5316455696202531,
          "recall": 0.1640625,
          "f1": 0.2507462686567164,
          "support": 256
        },
        "7": {
          "precision": 0.39672131147540984,
          "recall": 0.5931372549019608,
          "f1": 0.4754420432220039,
          "support": 204
        },
        "8": {
          "precision": 0.15384615384615385,
          "recall": 0.14736842105263157,
          "f1": 0.15053763440860213,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          87,
          6,
          17,
          1,
          6,
          19,
          9,
          4,
          11
        ],
        [
          21,
          51,
          34,
          11,
          13,
          46,
          2,
          9,
          18
        ],
        [
          15,
          2,
          107,
          2,
          1,
          1,
          1,
          0,
          14
        ],
        [
          14,
          0,
          15,
          53,
          1,
          9,
          0,
          9,
          1
        ],
        [
          11,
          13,
          4,
          2,
          29,
          36,
          0,
          1,
          13
        ],
        [
          4,
          16,
          8,
          1,
          3,
          66,
          1,
          14,
          11
        ],
        [
          11,
          9,
          8,
          2,
          1,
          17,
          42,
          122,
          44
        ],
        [
          0,
          1,
          11,
          0,
          0,
          12,
          17,
          121,
          42
        ],
        [
          13,
          24,
          41,
          4,
          8,
          40,
          7,
          25,
          28
        ]
      ],
      "processing_time_seconds": 82.8668041229248,
      "samples_per_second": 18.016864724085178,
      "enhancements_used": {
        "chain_of_thought": true,
        "multiple_sampling": false,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 3,
      "total_samples": 1493,
      "valid_predictions": 1490,
      "invalid_predictions": 3,
      "overall_accuracy": 0.3718120805369127,
      "correct_predictions": 554,
      "average_confidence": 0.9774049217002239,
      "per_class_metrics": {
        "0": {
          "precision": 0.29473684210526313,
          "recall": 0.7044025157232704,
          "f1": 0.4155844155844156,
          "support": 159
        },
        "1": {
          "precision": 0.29375,
          "recall": 0.22926829268292684,
          "f1": 0.25753424657534246,
          "support": 205
        },
        "2": {
          "precision": 0.4541284403669725,
          "recall": 0.6923076923076923,
          "f1": 0.5484764542936288,
          "support": 143
        },
        "3": {
          "precision": 0.7454545454545455,
          "recall": 0.40594059405940597,
          "f1": 0.5256410256410257,
          "support": 101
        },
        "4": {
          "precision": 0.449438202247191,
          "recall": 0.3669724770642202,
          "f1": 0.4040404040404041,
          "support": 109
        },
        "5": {
          "precision": 0.3584905660377358,
          "recall": 0.3064516129032258,
          "f1": 0.33043478260869563,
          "support": 124
        },
        "6": {
          "precision": 0.5037037037037037,
          "recall": 0.265625,
          "f1": 0.34782608695652173,
          "support": 256
        },
        "7": {
          "precision": 0.4482758620689655,
          "recall": 0.44607843137254904,
          "f1": 0.44717444717444726,
          "support": 204
        },
        "8": {
          "precision": 0.125,
          "recall": 0.09523809523809523,
          "f1": 0.1081081081081081,
          "support": 189
        }
      },
      "confusion_matrix": [
        [
          112,
          6,
          14,
          1,
          6,
          3,
          10,
          2,
          5
        ],
        [
          63,
          47,
          28,
          7,
          23,
          19,
          3,
          7,
          8
        ],
        [
          28,
          2,
          99,
          1,
          1,
          0,
          2,
          0,
          10
        ],
        [
          32,
          3,
          13,
          41,
          0,
          4,
          0,
          7,
          1
        ],
        [
          27,
          22,
          2,
          0,
          40,
          13,
          0,
          0,
          5
        ],
        [
          22,
          34,
          6,
          0,
          6,
          38,
          7,
          8,
          3
        ],
        [
          36,
          7,
          6,
          2,
          2,
          5,
          68,
          71,
          59
        ],
        [
          15,
          9,
          11,
          0,
          0,
          8,
          35,
          91,
          35
        ],
        [
          45,
          30,
          39,
          3,
          11,
          16,
          10,
          17,
          18
        ]
      ],
      "processing_time_seconds": 59.25151324272156,
      "samples_per_second": 25.197668688797577,
      "enhancements_used": {
        "chain_of_thought": false,
        "multiple_sampling": true,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 3,
      "total_samples": 1493,
      "valid_predictions": 1490,
      "invalid_predictions": 3,
      "overall_accuracy": 0.39932885906040266,
      "correct_predictions": 595,
      "average_confidence": 0.9715883668903802,
      "per_class_metrics": {
        "0": {
          "precision": 0.4885057471264368,
          "recall": 0.5345911949685535,
          "f1": 0.5105105105105106,
          "support": 159
        },
        "1": {
          "precision": 0.4297520661157025,
          "recall": 0.25365853658536586,
          "f1": 0.3190184049079755,
          "support": 205
        },
        "2": {
          "precision": 0.4448979591836735,
          "recall": 0.7622377622377622,
          "f1": 0.5618556701030928,
          "support": 143
        },
        "3": {
          "precision": 0.6708860759493671,
          "recall": 0.5196078431372549,
          "f1": 0.5856353591160222,
          "support": 102
        },
        "4": {
          "precision": 0.47692307692307695,
          "recall": 0.28703703703703703,
          "f1": 0.3583815028901734,
          "support": 108
        },
        "5": {
          "precision": 0.2791666666666667,
          "recall": 0.5403225806451613,
          "f1": 0.36813186813186816,
          "support": 124
        },
        "6": {
          "precision": 0.5569620253164557,
          "recall": 0.17254901960784313,
          "f1": 0.2634730538922156,
          "support": 255
        },
        "7": {
          "precision": 0.4031746031746032,
          "recall": 0.6225490196078431,
          "f1": 0.4894026974951831,
          "support": 204
        },
        "8": {
          "precision": 0.1569767441860465,
          "recall": 0.14210526315789473,
          "f1": 0.14917127071823202,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          85,
          5,
          18,
          3,
          6,
          18,
          9,
          4,
          11
        ],
        [
          22,
          52,
          34,
          11,
          15,
          42,
          2,
          9,
          18
        ],
        [
          14,
          2,
          109,
          2,
          1,
          1,
          1,
          0,
          13
        ],
        [
          14,
          0,
          12,
          53,
          1,
          12,
          0,
          9,
          1
        ],
        [
          10,
          13,
          4,
          2,
          31,
          35,
          0,
          0,
          13
        ],
        [
          5,
          13,
          8,
          1,
          3,
          67,
          1,
          16,
          10
        ],
        [
          11,
          9,
          8,
          2,
          0,
          16,
          44,
          122,
          43
        ],
        [
          0,
          2,
          11,
          0,
          0,
          11,
          17,
          127,
          36
        ],
        [
          13,
          25,
          41,
          5,
          8,
          38,
          5,
          28,
          27
        ]
      ],
      "processing_time_seconds": 99.5939462184906,
      "samples_per_second": 14.990870998570893,
      "enhancements_used": {
        "chain_of_thought": true,
        "multiple_sampling": true,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 5,
      "total_samples": 1493,
      "valid_predictions": 1493,
      "invalid_predictions": 0,
      "overall_accuracy": 0.36771600803750837,
      "correct_predictions": 549,
      "average_confidence": 1.0,
      "per_class_metrics": {
        "0": {
          "precision": 0.3257328990228013,
          "recall": 0.625,
          "f1": 0.4282655246252677,
          "support": 160
        },
        "1": {
          "precision": 0.3076923076923077,
          "recall": 0.17560975609756097,
          "f1": 0.2236024844720497,
          "support": 205
        },
        "2": {
          "precision": 0.4351464435146444,
          "recall": 0.7272727272727273,
          "f1": 0.5445026178010471,
          "support": 143
        },
        "3": {
          "precision": 0.5,
          "recall": 0.4803921568627451,
          "f1": 0.49,
          "support": 102
        },
        "4": {
          "precision": 0.36423841059602646,
          "recall": 0.5045871559633027,
          "f1": 0.423076923076923,
          "support": 109
        },
        "5": {
          "precision": 0.3764705882352941,
          "recall": 0.25806451612903225,
          "f1": 0.3062200956937799,
          "support": 124
        },
        "6": {
          "precision": 0.5491803278688525,
          "recall": 0.26171875,
          "f1": 0.35449735449735453,
          "support": 256
        },
        "7": {
          "precision": 0.5096774193548387,
          "recall": 0.3872549019607843,
          "f1": 0.44011142061281333,
          "support": 204
        },
        "8": {
          "precision": 0.1232876712328767,
          "recall": 0.14210526315789473,
          "f1": 0.13202933985330073,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          100,
          6,
          16,
          2,
          18,
          4,
          7,
          1,
          6
        ],
        [
          46,
          36,
          30,
          22,
          36,
          11,
          2,
          5,
          17
        ],
        [
          22,
          1,
          104,
          3,
          1,
          0,
          2,
          0,
          10
        ],
        [
          23,
          0,
          15,
          49,
          4,
          3,
          0,
          7,
          1
        ],
        [
          16,
          14,
          2,
          4,
          55,
          8,
          0,
          0,
          10
        ],
        [
          22,
          29,
          9,
          2,
          8,
          32,
          5,
          7,
          10
        ],
        [
          35,
          4,
          10,
          8,
          5,
          5,
          67,
          40,
          82
        ],
        [
          11,
          5,
          14,
          0,
          3,
          5,
          31,
          79,
          56
        ],
        [
          32,
          22,
          39,
          8,
          21,
          17,
          8,
          16,
          27
        ]
      ],
      "processing_time_seconds": 47.878151178359985,
      "samples_per_second": 31.18332607368531,
      "enhancements_used": {
        "chain_of_thought": false,
        "multiple_sampling": false,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 5,
      "total_samples": 1493,
      "valid_predictions": 1493,
      "invalid_predictions": 0,
      "overall_accuracy": 0.39048894842598797,
      "correct_predictions": 583,
      "average_confidence": 1.0,
      "per_class_metrics": {
        "0": {
          "precision": 0.3772893772893773,
          "recall": 0.64375,
          "f1": 0.47575057736720555,
          "support": 160
        },
        "1": {
          "precision": 0.41732283464566927,
          "recall": 0.25853658536585367,
          "f1": 0.31927710843373497,
          "support": 205
        },
        "2": {
          "precision": 0.4112903225806452,
          "recall": 0.7132867132867133,
          "f1": 0.5217391304347826,
          "support": 143
        },
        "3": {
          "precision": 0.711864406779661,
          "recall": 0.4117647058823529,
          "f1": 0.5217391304347826,
          "support": 102
        },
        "4": {
          "precision": 0.40540540540540543,
          "recall": 0.27522935779816515,
          "f1": 0.3278688524590164,
          "support": 109
        },
        "5": {
          "precision": 0.30978260869565216,
          "recall": 0.4596774193548387,
          "f1": 0.3701298701298701,
          "support": 124
        },
        "6": {
          "precision": 0.52,
          "recall": 0.203125,
          "f1": 0.29213483146067415,
          "support": 256
        },
        "7": {
          "precision": 0.41471571906354515,
          "recall": 0.6078431372549019,
          "f1": 0.493041749502982,
          "support": 204
        },
        "8": {
          "precision": 0.15503875968992248,
          "recall": 0.10526315789473684,
          "f1": 0.12539184952978055,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          103,
          6,
          17,
          0,
          10,
          9,
          8,
          2,
          5
        ],
        [
          37,
          53,
          35,
          7,
          16,
          34,
          2,
          8,
          13
        ],
        [
          26,
          3,
          102,
          2,
          1,
          0,
          1,
          0,
          8
        ],
        [
          25,
          0,
          13,
          42,
          4,
          10,
          0,
          7,
          1
        ],
        [
          17,
          14,
          3,
          2,
          30,
          26,
          1,
          0,
          16
        ],
        [
          12,
          16,
          9,
          1,
          4,
          57,
          5,
          14,
          6
        ],
        [
          22,
          9,
          8,
          1,
          2,
          10,
          52,
          114,
          38
        ],
        [
          7,
          2,
          18,
          0,
          0,
          8,
          23,
          124,
          22
        ],
        [
          24,
          24,
          43,
          4,
          7,
          30,
          8,
          30,
          20
        ]
      ],
      "processing_time_seconds": 117.25468230247498,
      "samples_per_second": 12.73296699699033,
      "enhancements_used": {
        "chain_of_thought": true,
        "multiple_sampling": false,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 5,
      "total_samples": 1493,
      "valid_predictions": 1491,
      "invalid_predictions": 2,
      "overall_accuracy": 0.3615023474178404,
      "correct_predictions": 539,
      "average_confidence": 0.9778672032193159,
      "per_class_metrics": {
        "0": {
          "precision": 0.31629392971246006,
          "recall": 0.61875,
          "f1": 0.4186046511627907,
          "support": 160
        },
        "1": {
          "precision": 0.2916666666666667,
          "recall": 0.17073170731707318,
          "f1": 0.21538461538461542,
          "support": 205
        },
        "2": {
          "precision": 0.4369747899159664,
          "recall": 0.7272727272727273,
          "f1": 0.5459317585301837,
          "support": 143
        },
        "3": {
          "precision": 0.5052631578947369,
          "recall": 0.4752475247524752,
          "f1": 0.4897959183673469,
          "support": 101
        },
        "4": {
          "precision": 0.35664335664335667,
          "recall": 0.46788990825688076,
          "f1": 0.40476190476190477,
          "support": 109
        },
        "5": {
          "precision": 0.39285714285714285,
          "recall": 0.2661290322580645,
          "f1": 0.31730769230769235,
          "support": 124
        },
        "6": {
          "precision": 0.5196850393700787,
          "recall": 0.2578125,
          "f1": 0.34464751958224543,
          "support": 256
        },
        "7": {
          "precision": 0.4935064935064935,
          "recall": 0.37254901960784315,
          "f1": 0.42458100558659223,
          "support": 204
        },
        "8": {
          "precision": 0.12442396313364056,
          "recall": 0.14285714285714285,
          "f1": 0.1330049261083744,
          "support": 189
        }
      },
      "confusion_matrix": [
        [
          99,
          6,
          16,
          2,
          18,
          3,
          9,
          1,
          6
        ],
        [
          47,
          35,
          29,
          23,
          35,
          11,
          2,
          5,
          18
        ],
        [
          23,
          1,
          104,
          2,
          1,
          0,
          2,
          0,
          10
        ],
        [
          22,
          0,
          15,
          48,
          4,
          3,
          0,
          7,
          2
        ],
        [
          18,
          16,
          2,
          4,
          51,
          8,
          0,
          0,
          10
        ],
        [
          21,
          29,
          10,
          2,
          8,
          33,
          5,
          7,
          9
        ],
        [
          37,
          4,
          9,
          7,
          4,
          5,
          66,
          43,
          81
        ],
        [
          15,
          6,
          13,
          0,
          3,
          3,
          34,
          76,
          54
        ],
        [
          31,
          23,
          40,
          7,
          19,
          18,
          9,
          15,
          27
        ]
      ],
      "processing_time_seconds": 65.3693790435791,
      "samples_per_second": 22.839439839327184,
      "enhancements_used": {
        "chain_of_thought": false,
        "multiple_sampling": true,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 5,
      "total_samples": 1493,
      "valid_predictions": 1489,
      "invalid_predictions": 4,
      "overall_accuracy": 0.4016118200134318,
      "correct_predictions": 598,
      "average_confidence": 0.9735840608909784,
      "per_class_metrics": {
        "0": {
          "precision": 0.3821428571428571,
          "recall": 0.6729559748427673,
          "f1": 0.48747152619589973,
          "support": 159
        },
        "1": {
          "precision": 0.4409448818897638,
          "recall": 0.2731707317073171,
          "f1": 0.3373493975903615,
          "support": 205
        },
        "2": {
          "precision": 0.4180327868852459,
          "recall": 0.7132867132867133,
          "f1": 0.5271317829457364,
          "support": 143
        },
        "3": {
          "precision": 0.7457627118644068,
          "recall": 0.43137254901960786,
          "f1": 0.546583850931677,
          "support": 102
        },
        "4": {
          "precision": 0.43661971830985913,
          "recall": 0.28703703703703703,
          "f1": 0.34636871508379885,
          "support": 108
        },
        "5": {
          "precision": 0.33513513513513515,
          "recall": 0.5040650406504065,
          "f1": 0.4025974025974026,
          "support": 123
        },
        "6": {
          "precision": 0.5483870967741935,
          "recall": 0.19921875,
          "f1": 0.2922636103151862,
          "support": 256
        },
        "7": {
          "precision": 0.40863787375415284,
          "recall": 0.6029411764705882,
          "f1": 0.4871287128712871,
          "support": 204
        },
        "8": {
          "precision": 0.17054263565891473,
          "recall": 0.1164021164021164,
          "f1": 0.13836477987421383,
          "support": 189
        }
      },
      "confusion_matrix": [
        [
          107,
          4,
          15,
          0,
          8,
          10,
          7,
          2,
          6
        ],
        [
          39,
          56,
          36,
          7,
          15,
          28,
          2,
          8,
          14
        ],
        [
          26,
          2,
          102,
          2,
          1,
          1,
          1,
          0,
          8
        ],
        [
          22,
          0,
          13,
          44,
          2,
          11,
          0,
          9,
          1
        ],
        [
          20,
          14,
          3,
          1,
          31,
          26,
          0,
          0,
          13
        ],
        [
          13,
          14,
          8,
          1,
          3,
          62,
          4,
          13,
          5
        ],
        [
          23,
          10,
          7,
          1,
          2,
          10,
          51,
          116,
          36
        ],
        [
          7,
          2,
          18,
          0,
          0,
          9,
          21,
          123,
          24
        ],
        [
          23,
          25,
          42,
          3,
          9,
          28,
          7,
          30,
          22
        ]
      ],
      "processing_time_seconds": 136.01122784614563,
      "samples_per_second": 10.977034937798406,
      "enhancements_used": {
        "chain_of_thought": true,
        "multiple_sampling": true,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 7,
      "total_samples": 1493,
      "valid_predictions": 1493,
      "invalid_predictions": 0,
      "overall_accuracy": 0.3737441393168118,
      "correct_predictions": 558,
      "average_confidence": 1.0,
      "per_class_metrics": {
        "0": {
          "precision": 0.3422818791946309,
          "recall": 0.6375,
          "f1": 0.445414847161572,
          "support": 160
        },
        "1": {
          "precision": 0.32335329341317365,
          "recall": 0.2634146341463415,
          "f1": 0.29032258064516125,
          "support": 205
        },
        "2": {
          "precision": 0.46798029556650245,
          "recall": 0.6643356643356644,
          "f1": 0.5491329479768786,
          "support": 143
        },
        "3": {
          "precision": 0.5512820512820513,
          "recall": 0.4215686274509804,
          "f1": 0.47777777777777775,
          "support": 102
        },
        "4": {
          "precision": 0.376,
          "recall": 0.43119266055045874,
          "f1": 0.4017094017094018,
          "support": 109
        },
        "5": {
          "precision": 0.40404040404040403,
          "recall": 0.3225806451612903,
          "f1": 0.35874439461883406,
          "support": 124
        },
        "6": {
          "precision": 0.532258064516129,
          "recall": 0.2578125,
          "f1": 0.34736842105263155,
          "support": 256
        },
        "7": {
          "precision": 0.43523316062176165,
          "recall": 0.4117647058823529,
          "f1": 0.42317380352644834,
          "support": 204
        },
        "8": {
          "precision": 0.13106796116504854,
          "recall": 0.14210526315789473,
          "f1": 0.13636363636363638,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          102,
          8,
          14,
          0,
          17,
          1,
          8,
          2,
          8
        ],
        [
          44,
          54,
          25,
          16,
          26,
          15,
          4,
          7,
          14
        ],
        [
          25,
          4,
          95,
          3,
          2,
          1,
          2,
          0,
          11
        ],
        [
          21,
          6,
          12,
          43,
          4,
          6,
          0,
          9,
          1
        ],
        [
          15,
          19,
          2,
          4,
          47,
          9,
          0,
          0,
          13
        ],
        [
          19,
          25,
          4,
          2,
          8,
          40,
          4,
          11,
          11
        ],
        [
          34,
          11,
          5,
          2,
          3,
          5,
          66,
          60,
          70
        ],
        [
          9,
          8,
          11,
          1,
          3,
          5,
          32,
          84,
          51
        ],
        [
          29,
          32,
          35,
          7,
          15,
          17,
          8,
          20,
          27
        ]
      ],
      "processing_time_seconds": 49.76764678955078,
      "samples_per_second": 29.999409180694283,
      "enhancements_used": {
        "chain_of_thought": false,
        "multiple_sampling": false,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 7,
      "total_samples": 1493,
      "valid_predictions": 1493,
      "invalid_predictions": 0,
      "overall_accuracy": 0.38513060951105155,
      "correct_predictions": 575,
      "average_confidence": 1.0,
      "per_class_metrics": {
        "0": {
          "precision": 0.52,
          "recall": 0.4875,
          "f1": 0.5032258064516129,
          "support": 160
        },
        "1": {
          "precision": 0.37583892617449666,
          "recall": 0.2731707317073171,
          "f1": 0.3163841807909605,
          "support": 205
        },
        "2": {
          "precision": 0.4151624548736462,
          "recall": 0.8041958041958042,
          "f1": 0.5476190476190476,
          "support": 143
        },
        "3": {
          "precision": 0.5473684210526316,
          "recall": 0.5098039215686274,
          "f1": 0.5279187817258884,
          "support": 102
        },
        "4": {
          "precision": 0.42857142857142855,
          "recall": 0.30275229357798167,
          "f1": 0.3548387096774194,
          "support": 109
        },
        "5": {
          "precision": 0.30344827586206896,
          "recall": 0.3548387096774194,
          "f1": 0.32713754646840143,
          "support": 124
        },
        "6": {
          "precision": 0.5392156862745098,
          "recall": 0.21484375,
          "f1": 0.30726256983240224,
          "support": 256
        },
        "7": {
          "precision": 0.4036363636363636,
          "recall": 0.5441176470588235,
          "f1": 0.4634655532359081,
          "support": 204
        },
        "8": {
          "precision": 0.13901345291479822,
          "recall": 0.1631578947368421,
          "f1": 0.15012106537530268,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          78,
          9,
          21,
          5,
          9,
          11,
          8,
          3,
          16
        ],
        [
          15,
          56,
          36,
          21,
          19,
          24,
          3,
          8,
          23
        ],
        [
          9,
          3,
          115,
          3,
          1,
          0,
          1,
          0,
          11
        ],
        [
          15,
          1,
          17,
          52,
          1,
          6,
          0,
          9,
          1
        ],
        [
          6,
          18,
          4,
          3,
          33,
          23,
          0,
          1,
          21
        ],
        [
          4,
          18,
          11,
          1,
          4,
          44,
          5,
          22,
          15
        ],
        [
          11,
          7,
          13,
          3,
          3,
          10,
          55,
          95,
          59
        ],
        [
          0,
          4,
          14,
          0,
          0,
          6,
          23,
          111,
          46
        ],
        [
          12,
          33,
          46,
          7,
          7,
          21,
          7,
          26,
          31
        ]
      ],
      "processing_time_seconds": 154.63345980644226,
      "samples_per_second": 9.655090184678125,
      "enhancements_used": {
        "chain_of_thought": true,
        "multiple_sampling": false,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 7,
      "total_samples": 1493,
      "valid_predictions": 1491,
      "invalid_predictions": 2,
      "overall_accuracy": 0.3735747820254863,
      "correct_predictions": 557,
      "average_confidence": 0.9796557120500782,
      "per_class_metrics": {
        "0": {
          "precision": 0.3333333333333333,
          "recall": 0.63125,
          "f1": 0.4362850971922246,
          "support": 160
        },
        "1": {
          "precision": 0.3151515151515151,
          "recall": 0.25365853658536586,
          "f1": 0.28108108108108104,
          "support": 205
        },
        "2": {
          "precision": 0.458128078817734,
          "recall": 0.6503496503496503,
          "f1": 0.5375722543352601,
          "support": 143
        },
        "3": {
          "precision": 0.5454545454545454,
          "recall": 0.4158415841584158,
          "f1": 0.47191011235955055,
          "support": 101
        },
        "4": {
          "precision": 0.3949579831932773,
          "recall": 0.43119266055045874,
          "f1": 0.4122807017543859,
          "support": 109
        },
        "5": {
          "precision": 0.4,
          "recall": 0.3387096774193548,
          "f1": 0.3668122270742357,
          "support": 124
        },
        "6": {
          "precision": 0.5483870967741935,
          "recall": 0.265625,
          "f1": 0.35789473684210527,
          "support": 256
        },
        "7": {
          "precision": 0.4427083333333333,
          "recall": 0.4166666666666667,
          "f1": 0.4292929292929293,
          "support": 204
        },
        "8": {
          "precision": 0.1330049261083744,
          "recall": 0.14285714285714285,
          "f1": 0.1377551020408163,
          "support": 189
        }
      },
      "confusion_matrix": [
        [
          101,
          8,
          14,
          1,
          15,
          2,
          9,
          2,
          8
        ],
        [
          48,
          52,
          27,
          16,
          24,
          13,
          4,
          8,
          13
        ],
        [
          27,
          4,
          93,
          2,
          2,
          1,
          2,
          0,
          12
        ],
        [
          22,
          3,
          12,
          42,
          4,
          8,
          0,
          9,
          1
        ],
        [
          16,
          18,
          2,
          4,
          47,
          10,
          0,
          0,
          12
        ],
        [
          18,
          26,
          4,
          2,
          8,
          42,
          4,
          9,
          11
        ],
        [
          34,
          13,
          5,
          2,
          2,
          5,
          68,
          59,
          68
        ],
        [
          8,
          8,
          11,
          1,
          3,
          8,
          29,
          85,
          51
        ],
        [
          29,
          33,
          35,
          7,
          14,
          16,
          8,
          20,
          27
        ]
      ],
      "processing_time_seconds": 69.61878204345703,
      "samples_per_second": 21.445362245321217,
      "enhancements_used": {
        "chain_of_thought": false,
        "multiple_sampling": true,
        "confidence_threshold": 0.6
      }
    },
    {
      "few_shot_count": 7,
      "total_samples": 1493,
      "valid_predictions": 1488,
      "invalid_predictions": 5,
      "overall_accuracy": 0.38911290322580644,
      "correct_predictions": 579,
      "average_confidence": 0.9742383512544802,
      "per_class_metrics": {
        "0": {
          "precision": 0.5384615384615384,
          "recall": 0.48427672955974843,
          "f1": 0.509933774834437,
          "support": 159
        },
        "1": {
          "precision": 0.3815789473684211,
          "recall": 0.28431372549019607,
          "f1": 0.3258426966292135,
          "support": 204
        },
        "2": {
          "precision": 0.4218181818181818,
          "recall": 0.8111888111888111,
          "f1": 0.5550239234449761,
          "support": 143
        },
        "3": {
          "precision": 0.5473684210526316,
          "recall": 0.5098039215686274,
          "f1": 0.5279187817258884,
          "support": 102
        },
        "4": {
          "precision": 0.4430379746835443,
          "recall": 0.32407407407407407,
          "f1": 0.37433155080213903,
          "support": 108
        },
        "5": {
          "precision": 0.3263888888888889,
          "recall": 0.3821138211382114,
          "f1": 0.352059925093633,
          "support": 123
        },
        "6": {
          "precision": 0.5252525252525253,
          "recall": 0.20392156862745098,
          "f1": 0.2937853107344633,
          "support": 255
        },
        "7": {
          "precision": 0.4072727272727273,
          "recall": 0.5490196078431373,
          "f1": 0.4676409185803758,
          "support": 204
        },
        "8": {
          "precision": 0.13274336283185842,
          "recall": 0.15789473684210525,
          "f1": 0.14423076923076922,
          "support": 190
        }
      },
      "confusion_matrix": [
        [
          77,
          9,
          20,
          5,
          8,
          11,
          8,
          3,
          18
        ],
        [
          12,
          58,
          35,
          22,
          19,
          22,
          3,
          8,
          25
        ],
        [
          9,
          3,
          116,
          2,
          1,
          0,
          1,
          0,
          11
        ],
        [
          13,
          1,
          18,
          52,
          3,
          6,
          0,
          9,
          0
        ],
        [
          5,
          19,
          4,
          3,
          35,
          21,
          0,
          1,
          20
        ],
        [
          4,
          19,
          10,
          1,
          4,
          47,
          4,
          18,
          16
        ],
        [
          11,
          8,
          12,
          3,
          2,
          9,
          52,
          98,
          60
        ],
        [
          0,
          3,
          14,
          0,
          0,
          6,
          23,
          112,
          46
        ],
        [
          12,
          32,
          46,
          7,
          7,
          22,
          8,
          26,
          30
        ]
      ],
      "processing_time_seconds": 175.8313283920288,
      "samples_per_second": 8.491092080424071,
      "enhancements_used": {
        "chain_of_thought": true,
        "multiple_sampling": true,
        "confidence_threshold": 0.6
      }
    }
  ],
  "analysis": {}
}