# Grievance Category Hierarchy Prediction

Data columns (total 8 columns):

| Index | Column Name             | Description                       |
|-------|-------------------------|-----------------------------------|
| 0     | CategoryCode            | the last level code               |
| 1     | remarks_text            | the response from Dept.           |
| 2     | subject_content_text    | the grievance                     |
| 3     | root_category_name      | the top-level category            |
| 4     | root_category_code      | the top-level category code       |
| 5     | total_no_of_stages      | the total stages in the hierarchy |
| 6     | hierarchy_order         | the order of the hierarchy        |
| 7     | category_hierarchy_code | the code of the hierarchy         |

```json
[
  {
    "CategoryV7": 2369.0,
    "remarks_text": "some random remark",
    "subject_content_text": "some random grievance",
    "root_category_name": "Labour and Employment",
    "root_category_code": 2173.0,
    "total_no_of_stages": 3,
    "hierarchy_order": [
      "Labour and Employment",
      "PF Withdrawal",
      "Others"
    ],
    "category_hierarchy_code": [
      2173,
      2343,
      2369
    ]
  },
  {
    "CategoryV7": 20493.0,
    "remarks_text": "Some random remark",
    "subject_content_text": "some random grievance",
    "root_category_name": "Department of Defence",
    "root_category_code": 6300.0,
    "total_no_of_stages": 3,
    "hierarchy_order": [
      "Department of Defence",
      "Canteen Stores Depot related",
      "Non entry to dependent"
    ],
    "category_hierarchy_code": [
      6300,
      6304,
      20493
    ]
  }
]
```