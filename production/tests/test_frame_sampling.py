from refinement.select_frame.sample_by_sharpness import choose_grouped_frames


def test_grouped_sampling_returns_exact_requested_count() -> None:
    names = [f"{index:05d}.png" for index in range(70)]
    sharpness = {name: float(index) for index, name in enumerate(names)}

    selected = choose_grouped_frames(names, sharpness, num_views=16)

    assert len(selected) == 16
    assert len(set(selected)) == 16
    assert selected == sorted(selected)
    assert selected[0] == "00003.png"
    assert selected[-1] == "00069.png"


def test_grouped_sampling_caps_count_at_available_frames() -> None:
    names = ["00000.png", "00001.png", "00002.png"]
    sharpness = {name: 1.0 for name in names}

    selected = choose_grouped_frames(names, sharpness, num_views=16)

    assert selected == names
