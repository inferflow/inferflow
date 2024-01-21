# Getting Started on Windows

This document contains instructions about building and running the Inferflow tools and service **on Windows**.

Before getting started, please make sure that:
1. The [**source codes of Inferflow**](https://github.com/inferflow/inferflow) has been cloned to the local machine.
2. Microsoft Visual Studio (2017 or a newer version) has been installed.
3. [**NVCC 12.2**](https://developer.nvidia.com/cuda-toolkit-archive) is installed **after** the installation of Visual Studio.

    If Visual Studio is installed later, you may have to reinstall NVCC 12.2. During the reinstallation, please choose the **custom installation** option and then choose **NOT** to install the driver by unchecking the "driver components" checkbox.

## Build

  Open one of the following solution files in ```build/vs_projects``` or ```build/vs2022_projects```:
  - __inferflow.sln__: The GPU version that supports GPU/CPU hybrid inference.
  - __inferflow_cpu.sln__: The CPU-only version.

  For the Debug configuration, executables are generated to ```bin/x64_Debug/```; while the output directory for the Release configuration is ```bin/x64_Release/```.

## Run the LLM Inferencing Tool (llm_inference.exe)

* Example-1: Load a tiny model and perform inference

  - **Step-1**: Download the model

  ```
  #> cd {inferflow-root-dir}/data/models/llama2.c/
  #> download.win.cmd
  ```

  Instead of running the above batch script, you can also manually download the model files and copy them to the above folder. The source URL and file names can be found from download.win.cmd.

  - **Step-2**: Run the **llm_inference** tool:

  ```
  #> cd {inferflow-root-dir}/bin/
  #> x64_Release/llm_inference.exe llm_inference.tiny.ini
  ```
  Please note that it is okay for llm_inference.exe and llm_inference.tiny.ini not being in the same folder (llm_inference.tiny.ini is in **bin/** and llm_inference.exe is in **bin/x64_Release/**).

* Example-2: Run the **llm_inference** tool to load a larger model for inference

  - **Step-1**: Edit configuration file **bin/inferflow_service.ini** to choose a model.

  In the "transformer_engine" section of bin/inferflow_service.ini, there are multiple lines starting with "```models = ```" or "```;models = ```".
  The lines starting with the "**;**" character are comments.
  To choose a model for inference, please uncomment the line corresponding to this model, and comment the lines of other models.
  By default, the **phi-2** model is selected.

  - **Step-2**: Download the selected model.
  ```
  #> cd {inferflow-root-dir}/data/models/{model-name}/
  #> download.win.cmd
  ```

  - **Step-3**: Edit configuration file **bin/llm_inference.ini** to choose or edit a query.

  In the configuration file, queries are organized into query lists. A query list can contain one or multiple queries.
  Different query lists are for different purposes. For example, ```query_list.decoder_only``` is for testing decoder-only models. Its detailed information can be configured in the ```query_list.decoder_only``` section.
  The starting line of this section is "```query_count = 1```", which means only one query is included in this query list.
  Among the following lines with key ```query1```, only one line is uncommented and therefore effective, whereas other lines (i.e., the lines starting with a "**;**" character) are commented.
  You can choose a query for testing by uncommenting this query and commenting all the other queries. You can, of course, add new queries or change the content of an existing query.

  - **Step-4**: Run the inference tool

  ```
  #> cd {inferflow-root-dir}/bin/
  #> x64_Release/llm_inference.exe
  ```

## Run the Inferflow Service (inferflow_service.exe)

* Start the Inferflow service:

  - **Step-1**: Edit the service configuration file (bin/inferflow_service.ini)

  - **Step-2**: Start the service:
  ```
  #> cd bin
  #> x64_Release\inferflow_service.exe
  ```
  Alternatively, you can also run the service from the ```x64_Release``` folder:
  ```
  #> cd bin\x64_Release
  #> inferflow_service.exe
  ```

## Test the Inferflow Service

Run an HTTP client, to interact with the Inferflow service via HTTP protocol to get inference results.

* **Option-1**. Run the Inferflow client tool: inferflow_client.exe

  - **Step-1**: Edit the configuration file (bin/inferflow_client.ini) to set the query text and options.

  - **Step-2**: Run the client tool to get inference results.
  ```
  #> cd bin
  #> x64_Release\inferflow_client.exe
  ```

* **Option-2**. Use a third-party REST client (e.g., the Chrome extension of ```Tabbed Postman```).

  - **URL**: ```http://localhost:8080``` (If you access the service from a different machine)
  
  - **HTTP method**: ```POST```
  
  - **Example body text**: ```{"text": "Write an article about the weather of Seattle.", "res_prefix": "", "decoding_alg": "sample.top_p", "random_seed": 1, "temperature": 0.7, "is_streaming_mode": 0}```

## Run Other Tools

(To be added)

