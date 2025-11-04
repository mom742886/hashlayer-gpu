{
  "targets": [
    {
      "target_name": "blake2b_miner_cuda",
      "sources": [
        "miner.cc",
        "kernels.cu",
        "blake2b.cu"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")"
      ],
      "defines": [
        "NAPI_DISABLE_CPP_EXCEPTIONS"
      ],
      "cflags!": [
        "-fno-exceptions"
      ],
      "cflags_cc!": [
        "-fno-exceptions"
      ],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "MACOSX_DEPLOYMENT_TARGET": "10.9",
        "OTHER_CPLUSPLUSFLAGS": [
          "-std=c++14",
          "-stdlib=libc++"
        ],
        "OTHER_LDFLAGS": [
          "-lc++"
        ]
      },
      "msvs_settings": {
        "VCCLCompilerTool": {
          "ExceptionHandling": 1,
          "LanguageStandard": "stdcpp14"
        }
      },
      "conditions": [
        [
          "OS=='mac'",
          {
            "variables": {
              "cuda_toolkit_dir%": "<!(if [ -d /usr/local/cuda ]; then echo /usr/local/cuda; elif [ -d /opt/cuda ]; then echo /opt/cuda; elif [ -d /Developer/NVIDIA/CUDA-* ]; then ls -d /Developer/NVIDIA/CUDA-* 2>/dev/null | tail -1 || echo /usr/local/cuda; else echo /usr/local/cuda; fi)"
            },
            "include_dirs": [
              "<(cuda_toolkit_dir)/include"
            ],
            "libraries": [
              "-L<(cuda_toolkit_dir)/lib",
              "-lcudart"
            ],
            "link_settings": {
              "libraries": [
                "<(cuda_toolkit_dir)/lib/libcudart.dylib"
              ],
              "library_dirs": [
                "<(cuda_toolkit_dir)/lib"
              ]
            }
          }
        ],
        [
          "OS=='linux'",
          {
            "variables": {
              "cuda_toolkit_dir%": "<!(if [ -d /usr/local/cuda ]; then echo /usr/local/cuda; elif [ -d /opt/cuda ]; then echo /opt/cuda; else echo /usr/local/cuda; fi)"
            },
            "include_dirs": [
              "<(cuda_toolkit_dir)/include"
            ],
            "libraries": [
              "-L<(cuda_toolkit_dir)/lib64",
              "-lcudart"
            ],
            "cflags": [
              "-std=c++14"
            ],
            "cflags_cc": [
              "-std=c++14"
            ],
            "link_settings": {
              "libraries": [
                "<(cuda_toolkit_dir)/lib64/libcudart.so"
              ],
              "library_dirs": [
                "<(cuda_toolkit_dir)/lib64"
              ]
            }
          }
        ],
        [
          "OS=='win'",
          {
            "variables": {
              "cuda_path%": "<!(for /f \"delims=\" %i in ('where nvcc 2^>nul') do @echo %~dpi..)"
            },
            "include_dirs": [
              "<(cuda_path)/include"
            ],
            "libraries": [
              "<(cuda_path)/lib/x64/cudart.lib"
            ],
            "library_dirs": [
              "<(cuda_path)/lib/x64"
            ]
          }
        ]
      ],
      "libraries": [
        "<!@(node -p \"require('node-addon-api').gyp\")"
      ],
      "rules": [
        {
          "rule_name": "cuda_compile",
          "extension": "cu",
          "inputs": [
            "<(cuda_toolkit_dir)/bin/nvcc"
          ],
          "outputs": [
            "<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o"
          ],
          "message": "Compiling CUDA file <(RULE_INPUT_PATH)",
          "process_outputs_as_sources": 1,
          "action": [
            "<(cuda_toolkit_dir)/bin/nvcc",
            "-gencode=arch=compute_50,code=sm_50",
            "-gencode=arch=compute_60,code=sm_60",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_89,code=sm_89",
            "-O3",
            "-use_fast_math",
            "-Xcompiler",
            "-fPIC",
            "-c",
            "<(RULE_INPUT_PATH)",
            "-o",
            "<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o"
          ],
          "conditions": [
            [
              "OS=='mac'",
              {
                "action": [
                  "<(cuda_toolkit_dir)/bin/nvcc",
                  "-gencode=arch=compute_50,code=sm_50",
                  "-gencode=arch=compute_60,code=sm_60",
                  "-gencode=arch=compute_70,code=sm_70",
                  "-gencode=arch=compute_75,code=sm_75",
                  "-gencode=arch=compute_80,code=sm_80",
                  "-gencode=arch=compute_86,code=sm_86",
                  "-gencode=arch=compute_89,code=sm_89",
                  "-O3",
                  "-use_fast_math",
                  "-Xcompiler",
                  "-fPIC",
                  "-c",
                  "<(RULE_INPUT_PATH)",
                  "-o",
                  "<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o"
                ]
              }
            ],
            [
              "OS=='linux'",
              {
                "action": [
                  "<(cuda_toolkit_dir)/bin/nvcc",
                  "-gencode=arch=compute_50,code=sm_50",
                  "-gencode=arch=compute_60,code=sm_60",
                  "-gencode=arch=compute_70,code=sm_70",
                  "-gencode=arch=compute_75,code=sm_75",
                  "-gencode=arch=compute_80,code=sm_80",
                  "-gencode=arch=compute_86,code=sm_86",
                  "-gencode=arch=compute_89,code=sm_89",
                  "-O3",
                  "-use_fast_math",
                  "-Xcompiler",
                  "-fPIC",
                  "-c",
                  "<(RULE_INPUT_PATH)",
                  "-o",
                  "<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o"
                ]
              }
            ],
            [
              "OS=='win'",
              {
                "action": [
                  "<(cuda_path)/bin/nvcc.exe",
                  "-gencode=arch=compute_50,code=sm_50",
                  "-gencode=arch=compute_60,code=sm_60",
                  "-gencode=arch=compute_70,code=sm_70",
                  "-gencode=arch=compute_75,code=sm_75",
                  "-gencode=arch=compute_80,code=sm_80",
                  "-gencode=arch=compute_86,code=sm_86",
                  "-gencode=arch=compute_89,code=sm_89",
                  "-O3",
                  "-use_fast_math",
                  "-Xcompiler",
                  "/MD",
                  "-c",
                  "<(RULE_INPUT_PATH)",
                  "-o",
                  "<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o"
                ]
              }
            ]
          ]
        }
      ]
    }
  ]
}
