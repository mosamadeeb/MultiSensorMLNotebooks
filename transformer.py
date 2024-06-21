from mathutils import Vector, Quaternion, Euler
import pandas as pd
import numpy as np

ACC_ONLY_POSITIONS = [
    "LH",
    "RWR",
    "HIP",
    "RKN_",
]

IMU_ONLY_POSITIONS = [
    "LLA",
]

SHOE_POSITIONS = [
    "L-SHOE Body",
    "R-SHOE Body",
]

ACC_TO_IMU_POSITIONS = {
    "LUA^": "LUA",
    "BACK": "BACK",
    "RUA_": "RUA",
}

POSITIONS = ACC_ONLY_POSITIONS + IMU_ONLY_POSITIONS + SHOE_POSITIONS + list(ACC_TO_IMU_POSITIONS.keys())


class VectorTransformer:
    translation: Vector
    rotation: Quaternion
    scale: Vector

    def transform(self, vectors: list[Vector]) -> list[Vector]:
        result = []
        for v in vectors:
            vn = v.copy()
            vn.rotate(self.rotation)
            vn += self.translation

            result.append(Vector([a1 * a2 for a1, a2 in zip(vn[:], self.scale)]))

        return result


class QuaternionTransformer:
    rotation: Quaternion

    def transform(self, quaternions: list[Quaternion]) -> list[Quaternion]:
        result = []
        for q in quaternions:
            qn = q.copy()
            qn.rotate(self.rotation)

            result.append(qn)

        return result


class Transformer:
    acc: VectorTransformer
    quaternion: QuaternionTransformer
    imu_acc: VectorTransformer
    imu_euler: QuaternionTransformer
    imu_others: dict[str, VectorTransformer]

    def convert_transformed_vectors(self, transformed_vectors, time_col=None, columns=None) -> pd.DataFrame:
        if time_col is None:
            # Actual time doesn't matter
            time_col = list(range(len(transformed_vectors[POSITIONS[0]][0])))

        if columns is None:
            count = 0
            for pos in POSITIONS:
                if pos not in transformed_vectors:
                    continue

                count += len(get_acc_columns(pos))
                count += sum(len(sublist) for sublist in get_imu_columns(pos) + get_imu_acc_columns(pos))

            columns = list(range(count))

        df_test_data_transformed = pd.DataFrame(np.nan, index=time_col, dtype="Int64", columns=columns)

        for pos in POSITIONS:
            if pos not in transformed_vectors:
                continue

            acc, imu = transformed_vectors[pos]

            if len(acc) == 0:
                acc = [Vector((np.nan, np.nan, np.nan))] * len(time_col)

            acc_columns = get_acc_columns(pos)
            for i, col in enumerate(get_column_names_from_df(df_test_data_transformed, acc_columns)):
                df_test_data_transformed[col] = [round(v[i]) if not pd.isna(v[i]) else None for v in acc]

            imu_columns = get_imu_columns(pos) + get_imu_acc_columns(pos)
            for sublist in imu_columns:
                if len(sublist) == 0:
                    continue

                name = ""
                if "AngVel" in sublist[0]:
                    name = "ang_vel"
                elif "Eu" in sublist[0]:
                    name = "eu"
                elif "magnetic" in sublist[0]:
                    name = "magnetic"
                elif "gyro" in sublist[0]:
                    name = "gyro"
                elif "acc" in sublist[0]:
                    name = "acc"
                elif "Quaternion" in sublist[0]:
                    name = "quaternion"
                else:
                    continue

                if name not in imu:
                    continue

                if len(imu[name]) == 0:
                    if name == "quaternion":
                        imu[name] = [Quaternion((np.nan, np.nan, np.nan, np.nan))] * len(time_col)
                    else:
                        imu[name] = [Vector((np.nan, np.nan, np.nan))] * len(time_col)

                for i, col in enumerate(get_column_names_from_df(df_test_data_transformed, sublist)):
                    df_test_data_transformed[col] = [round(v[i]) if not pd.isna(v[i]) else None for v in imu[name]]

        df_test_data_transformed.index.name = "1 MILLISEC"
        df_test_data_transformed = df_test_data_transformed.astype("Int64")

        return df_test_data_transformed

    def transform_all(self, extracted_vectors, time_col=None, columns=None):
        result = dict()

        for pos in POSITIONS:
            if pos not in extracted_vectors:
                continue

            # Test data
            acc, imu = extracted_vectors[pos]

            # Apply the transformations
            transformed_acc = self.acc.transform(acc)
            transformed_imu_vectors = dict()

            for imu_col in imu:
                vectors = imu[imu_col]

                if imu_col == "quaternion":
                    transformed_imu_vectors[imu_col] = self.quaternion.transform(vectors)
                elif imu_col == "eu":
                    vectors: list[Quaternion] = [Euler((v * np.pi / 180)[:]).to_quaternion() for v in vectors]
                    transformed_imu_vectors[imu_col] = [
                        Vector(q.to_euler()[:]) * 180 / np.pi for q in self.quaternion.transform(vectors)
                    ]
                elif imu_col == "acc":
                    transformed_imu_vectors[imu_col] = self.imu_acc.transform(vectors)
                elif imu_col in ["ang_vel", "gyro", "magnetic"]:
                    transformed_imu_vectors[imu_col] = (
                        self.imu_others[imu_col].transform(vectors) if imu_col in self.imu_others else vectors
                    )

            result[pos] = [transformed_acc, transformed_imu_vectors]

        converted = self.convert_transformed_vectors(result, time_col, columns)

        return converted


def get_acc_columns(position: str) -> list[str]:
    """Returns 3 column names, for x, y, z

    Args:
        position (str): Position to get columns for

    Returns:
        list[str]: X, Y, Z columns for the given position
    """

    if position in ACC_ONLY_POSITIONS + list(ACC_TO_IMU_POSITIONS.keys()):
        return [f"Acc {position} acc{axis}" for axis in "XYZ"]
    elif position in IMU_ONLY_POSITIONS:
        return [f"IMU {position} acc{axis}" for axis in "XYZ"]
    elif position in SHOE_POSITIONS:
        return [f"IMU {position}_A{axis}" for axis in "xyz"]
    return []


def get_imu_acc_columns(position: str) -> list[list[str]]:
    if position in ACC_TO_IMU_POSITIONS:
        return [[f"IMU {ACC_TO_IMU_POSITIONS[position]} acc{axis}" for axis in "XYZ"]]

    return []


def get_imu_columns(position: str) -> list[list[str]]:
    """Returns a list of column name tuples for the given position

    Args:
        position (str): Position to get columns for

    Returns:
        list[list[str]]: List of column name tuples
    """

    if position in ACC_ONLY_POSITIONS:
        return []

    if position in SHOE_POSITIONS:
        shoe, place = position.split(" ")

        return [
            [f"IMU {shoe} Eu{axis}" for axis in "XYZ"],
            [f"IMU {shoe} AngVel{place}Frame{axis}" for axis in "XYZ"],
            # Ignore Compass column
        ]

    if position in ACC_TO_IMU_POSITIONS:
        position = ACC_TO_IMU_POSITIONS[position]

    return [
        [f"IMU {position} gyro{axis}" for axis in "XYZ"],
        [f"IMU {position} magnetic{axis}" for axis in "XYZ"],
        [f"IMU {position} Quaternion{axis}" for axis in range(1, 5)],
    ]


def get_column_names_from_df(df: pd.DataFrame, cols: list[str]) -> list[str]:
    # Get columns from df that that contain these columns as substring
    columns = [col for col in df.columns if any(named_col in col for named_col in cols)]

    # Sort the columns by X, Y, Z at the end of their names (or 1, 2, 3, 4 if quaternion)
    columns.sort(key=lambda x: x[-1])

    return columns


def get_columns_from_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(df, columns=get_column_names_from_df(df, cols))


def get_vector_list(df: pd.DataFrame, cols: list[str]) -> list[Vector]:
    df_cols = get_columns_from_df(df, cols)
    return [Vector(row.tolist()) for _, row in df_cols.iterrows()]


def get_quaternion_list(df: pd.DataFrame, cols: list[str]) -> list[Quaternion]:
    df_cols = get_columns_from_df(df, cols)
    return [Quaternion(row.tolist()) for _, row in df_cols.iterrows()]


def extract_time(df: pd.DataFrame):
    return df.index.tolist()


def extract_vectors(df: pd.DataFrame) -> dict[str, tuple[list[Vector], dict[str, list[Vector]]]]:
    result = dict()

    for pos in POSITIONS:
        acc_columns = get_acc_columns(pos)

        if len(acc_columns) == 0:
            continue

        # Create a list of vectors from the columns
        acc_vectors = get_vector_list(df, acc_columns)

        imu_columns = get_imu_columns(pos) + get_imu_acc_columns(pos)

        imu_vectors_dict = dict()
        for sublist in imu_columns:
            if len(sublist) == 0:
                continue

            if "Quaternion" in sublist[0]:
                # Create a list of quaternions from the columns
                imu_vectors_dict["quaternion"] = get_quaternion_list(df, sublist)
            else:
                # Create a list of vectors from the columns

                name = ""
                if "AngVel" in sublist[0]:
                    name = "ang_vel"
                elif "Eu" in sublist[0]:
                    name = "eu"
                elif "magnetic" in sublist[0]:
                    name = "magnetic"
                elif "gyro" in sublist[0]:
                    name = "gyro"
                elif "acc" in sublist[0]:
                    name = "acc"
                else:
                    continue

                imu_vectors_dict[name] = get_vector_list(df, sublist)

        result[pos] = (acc_vectors, imu_vectors_dict)

    return result


def read_dataset(dataset_path, is_new_format=False):
    df_dataset = pd.read_csv(dataset_path, sep="," if is_new_format else " ")

    # Set time column as index for dataset
    df_dataset.set_index("1 MILLISEC", inplace=True)

    return df_dataset


def get_static_pose_data(dataset_path):
    df_dataset = read_dataset(dataset_path)
    static_pose_data = extract_vectors(df_dataset)
    return static_pose_data


def transformer(test_data: pd.DataFrame, dataset_path="dataset_static_pose.csv") -> Transformer:
    dataset_vector_dict = get_static_pose_data(dataset_path)
    test_data_vector_dict = extract_vectors(test_data)

    def is_nan_vector(vector):
        return any(map(lambda x: pd.isna(x), vector))

    def test_range(data: list[Vector]):
        return [v for v in data if not is_nan_vector(v)]

    def calc_translation_difference(v1: Vector, v2: Vector) -> Vector:
        return v2 - v1

    def calc_rotation_difference(v1: Vector, v2: Vector) -> Quaternion:
        return v1.normalized().rotation_difference(v2.normalized())

    def calc_scale_difference(v1: Vector, v2: Vector) -> list[float]:
        # Element-wise scaling, not by magnitude since the model won't care for that
        return [a2 / a1 for a1, a2 in zip(v1, v2)]

    def transform_vectors(test_data: list[Vector], dataset_data: list[Vector]) -> VectorTransformer:
        test_data_range = test_range(test_data)

        if len(test_data_range) == 0:
            return test_data

        test_mean = sum(test_data_range, Vector()) / len(test_data_range)
        print(f"test mean: {test_mean}")

        dataset_mean = sum(dataset_data, Vector()) / len(dataset_data)
        print(f"* dataset mean: {dataset_mean}")

        print()

        translation_difference = calc_translation_difference(test_mean, dataset_mean)
        print(f"translation difference: {translation_difference}")

        quaternion_difference = calc_rotation_difference(test_mean, dataset_mean)
        print(f"quaternion difference: {quaternion_difference}")

        transformed: list[Vector] = []
        for v in test_data:
            vn = v.copy()
            vn.rotate(quaternion_difference)
            vn += translation_difference

            transformed.append(vn)

        temp_transformed = test_range(transformed)
        transformed_mean = sum(temp_transformed, Vector()) / len(temp_transformed)
        print(f"* transformed test mean: {transformed_mean}")

        print()

        # Scale after rotation and translation
        scale_difference = calc_scale_difference(transformed_mean, dataset_mean)
        print(f"scale difference: {scale_difference}")
        transformed = [Vector([a1 * a2 for a1, a2 in zip(a[:], scale_difference)]) for a in transformed]

        temp_transformed = test_range(transformed)
        transformed_mean = sum(temp_transformed, Vector()) / len(temp_transformed)
        print(f"** transformed scaled test mean: {transformed_mean}")

        result = VectorTransformer()
        result.translation = translation_difference
        result.rotation = quaternion_difference
        result.scale = scale_difference

        return result

    def quaternion_mean(quats: list[Quaternion]) -> Quaternion:
        identity = Quaternion()
        weight = 1 / len(quats)
        avg = Quaternion()

        for q in quats:
            avg.rotate(identity.slerp(q.normalized(), weight))

        return avg.normalized()

    def transform_quaternions(
        test_data: list[Quaternion], dataset_data: list[Quaternion], is_euler=False
    ) -> QuaternionTransformer:
        test_data_range = test_range(test_data)

        if len(test_data_range) == 0:
            return test_data

        test_mean = quaternion_mean(test_data_range)
        print(f"test mean: {test_mean.to_euler() if is_euler else test_mean}")

        dataset_mean = quaternion_mean(dataset_data)
        print(f"* dataset mean: {dataset_mean.to_euler() if is_euler else dataset_mean}")

        print()

        # Rotation difference
        quaternion_difference = calc_rotation_difference(test_mean, dataset_mean)
        print(f"quaternion difference: {quaternion_difference.to_euler() if is_euler else quaternion_difference}")

        transformed: list[Quaternion] = []
        for v in test_data:
            vn = v.copy()
            vn.rotate(quaternion_difference)

            transformed.append(vn)

        temp_transformed = test_range(transformed)
        transformed_mean = quaternion_mean(temp_transformed)
        print(f"* transformed test mean: {transformed_mean.to_euler() if is_euler else transformed_mean}")

        result = QuaternionTransformer()
        result.rotation = quaternion_difference

        return result

    for pos in POSITIONS:
        if not (pos in dataset_vector_dict and pos in test_data_vector_dict):
            continue

        # Test data
        acc, imu = test_data_vector_dict[pos]

        print(pos)

        # Apply the transformations
        acc_transformation = transform_vectors(acc, dataset_vector_dict[pos][0])
        imu_transformations = dict()

        for imu_col in imu:
            vectors = imu[imu_col]

            if imu_col == "quaternion":
                print("\nQuaternion")
                print("------------")
                quat_transformation = transform_quaternions(vectors, dataset_vector_dict[pos][1]["quaternion"])
            elif imu_col == "eu":
                vectors: list[Quaternion] = [Euler((v * np.pi / 180)[:]).to_quaternion() for v in vectors]
                dataset_vectors = [
                    Euler((v * np.pi / 180)[:]).to_quaternion() for v in dataset_vector_dict[pos][1]["eu"]
                ]

                print("\nEuler")
                print("------------")
                euler_transformation = transform_quaternions(vectors, dataset_vectors, is_euler=True)
            elif imu_col == "acc":
                # This only happens when the position has both an accelerometer and an IMU
                # In this case, we should transform the data using the mean of the IMU from the data set

                # The imu test data in this case is the same as the acc test data, because in the end
                # we only have 1 accelerometer for each position

                print(f"\nIMU Acc {pos}")
                print("------------")
                imu_acc_transformation = transform_vectors(vectors, dataset_vector_dict[pos][1]["acc"])
            elif imu_col in ["ang_vel", "gyro", "magnetic"]:
                print(f"\n{imu_col}")
                print("------------")
                imu_transformations[imu_col] = transform_vectors(vectors, dataset_vector_dict[pos][1][imu_col])

        print("--------------------------------\n")

    result = Transformer()
    result.acc = acc_transformation
    result.quaternion = quat_transformation
    result.imu_acc = imu_acc_transformation
    result.imu_euler = euler_transformation
    result.imu_others = imu_transformations

    return result


def transform_file(test_path, test_range_path, dataset_path=None):
    test_data = read_dataset(test_path, True)
    test_range_data = read_dataset(test_range_path)

    data_transformer = transformer(test_range_data, dataset_path) if dataset_path else transformer(test_range_data)
    transformed = data_transformer.transform_all(
        extract_vectors(test_data), test_data.index.tolist(), test_data.columns
    )

    transformed.to_csv(test_path + ".transformed.csv", index=True, sep=" ", na_rep="NaN")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} TEST_PATH TEST_RANGE_PATH [DATASET_PATH]")

    transform_file(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None)
