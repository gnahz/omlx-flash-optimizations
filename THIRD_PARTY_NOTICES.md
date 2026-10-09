# Third-party provenance

Repository additions are distributed under [Apache-2.0](LICENSE). The patch series modifies third-party source; original source headers, copyright notices and applicable licenses remain in force. Attribution here does not imply authorship of upstream algorithms or weights.

| Material | Fixed source | Notice |
|---|---|---|
| DeepSeek oMLX base | [jundot/omlx](https://github.com/jundot/omlx/tree/d4298ad1811bdd1b565660263380b52fc4ff71f0) | Apache-2.0; source license retained in the prepared checkout |
| DeepSeek MLX base and stock helper asset | [ml-explore/mlx](https://github.com/ml-explore/mlx/tree/59d600b5e64c238427d0f8d897ab7c682ef4d3d2) | MIT, Copyright © 2023 Apple Inc.; [license copy](docs/licenses/MLX-MIT.txt) |
| DeepSeek dependency patches | [tacos8me/m5-ultra](https://github.com/tacos8me/m5-ultra/tree/4e4d5a81f8135249815a3f284c0ed5bbd1faee58) | Source revision `4e4d5a81f8135249815a3f284c0ed5bbd1faee58`; original source notices retained |
| Qwen oMLX base | [jundot/omlx v0.7.0](https://github.com/jundot/omlx/tree/4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40) | Apache-2.0 |
| Qwen community patches | [sethforprivacy/Qwen3.8-Flash-Next-M5-Ultra-oMLX](https://github.com/sethforprivacy/Qwen3.8-Flash-Next-M5-Ultra-oMLX/tree/050233efcd0a765de9963ab21a01df0723a50530) | Revision `050233efcd0a765de9963ab21a01df0723a50530`; retain the recipe's original credits |
| Qwen kernel dependency | [oMLX PR #4030](https://github.com/jundot/omlx/pull/4030) | Original PR authorship retained |
| Qwen core fixtures | Pinned oMLX benchmark-corpus excerpts | Apache-2.0; original corpus/source attribution applies |
| DeepSeek model source | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/2cba9e42aa026125f3ed06c6d98c1db82f7ca027) | MIT, Copyright (c) 2023 DeepSeek; [license copy](docs/licenses/DeepSeek-MIT.txt), also preserved in the published weights |
| Qwen weights | [mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp/tree/e171af86f499f1855b0fb71d781105e8dd609610) | Downloaded from upstream, not redistributed here; preserve upstream model license |

DeepSeek uses the existing sourced quantization recipe; the author's additional work is exact metadata compression, the direct dictionary prefill path, qualified engine integration and cache persistence. Qwen uses community kernels/lookup and upstream oQ6e weights; the author's additions are policy safety and prefix-tail integration. No third-party benchmark numbers are presented as the author's measurements.

Original dependency licenses (including nanobind and build libraries) remain with their fetched packages. The stock MLX auxiliary asset is fetched from a pinned public wheel with hash verification, rather than redistributed in Git.
