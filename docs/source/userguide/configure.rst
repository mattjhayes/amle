#########
Configure
#########

The dataset class provides methods available through *project_policy.yaml*
to manipulate the ingested data so that it is suitable for processing.

*****************
Column Operations
*****************

Here are operations that can be performed on the dataset columns:

set_output_columns
==================

Sets what columns are used as output data from dataset
(i.e. what columns contain the expected answer(s)
Pass it a list of output column names

  Example:

  .. code-block:: text

    - set_output_columns:
        - Iris-setosa
        - Iris-versicolor
        - Iris-virginica
