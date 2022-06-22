# RL for Truss Layout

This is the AI+X course project by Siyang Wu and Zimeng Song, based on Xingcheng Yao's Alpha-truss and directed by Yi Wu, Xianzhong Zhao, and Ruifeng Luo.



## Stage 1

Generating truss layouts in stage 1.

```
python ./Stage1/main_uct.py
```

Change output's format to input for stage 2.

```
python ./Stage1/noise_input_permutation_format_transfer.py
```



## Stage 2

Use RL to search for lighter truss layouts.

```
python ./Stage2/main.py
```

For input with noise, use python files under folder `./Stage2/noise`.



## Results

| Points number | Alpha-truss | MCTS | Our Stage 1 | Our stage 2 (training 10 bar) | Our embedding method 1 (training on noise, testing on 10 bar) | Our embedding method 2 autoregression (training on noise, testing on 10 bar) |
| ------------- | ----------- | ---- | ----------- | ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 6             | 2305        | 2354 | 2184        | 2113.6                        | 2110.2                                                       | 2120                                                         |
| 7             | 3344        | 2335 | 1912.6      | 1813.7                        | 2110.1                                                       | 1745                                                         |
| 8             | 2994        | 2568 | 2520        | 2356                          | 2284.4                                                       | 2270.6                                                       |
| 9             | 2988        | 2582 | 2524        | 2174.8                        | 1985.4                                                       | 1945.1                                                       |