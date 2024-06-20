def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # 입력 텐서 x는 3차원이어야 함 [배치 크기, 시퀀스 길이, 특징 차원]
    assert n_state % hparams.n_head == 0  # n_state는 헤드 수의 배수여야 함
    if past is not None:
        assert past.shape.ndims == 5  # 과거 텐서는 5차원이어야 함 [배치 크기, 2, 헤드 수, 시퀀스 길이, 특징 차원], 여기서 2는 [k, v]를 의미함

    def split_heads(x):
        # [배치 크기, 시퀀스 길이, 특징 차원]에서 [배치 크기, 헤드 수, 시퀀스 길이, 특징 차원]으로 변환
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # split_heads의 역변환
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))


    def mask_attn_weights(w):
        # w의 형태는 [배치 크기, 헤드 수, 대상 시퀀스 길이, 소스 시퀀스 길이]이며, 정보는 소스에서 대상으로 흐름
        _, _, nd, ns = shape_list(w)  # w의 형상을 가져옴
        b = attention_mask(nd, ns, dtype=w.dtype)  # 하삼각형 마스크를 생성 (하단 삼각형에 1, 나머지에 0)
        b = tf.reshape(b, [1, 1, nd, ns])  # 마스크 b의 형상을 [1, 1, nd, ns]로 재형성
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)  # 마스크를 적용하여 미래 정보가 사용되지 않도록 함
        return w  # 마스킹된 가중치를 반환
