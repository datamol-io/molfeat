import os
import shutil
import unittest as ut
import datamol as dm
import pandas as pd
import torch
import numpy as np
import shelve
import tempfile
import h5py
import joblib
from molfeat.utils import datatype
from molfeat.utils import commons
from molfeat.utils.cache import CacheList, DataCache
from molfeat.utils.cache import FileCache
from molfeat.trans.fp import FPVecTransformer


def utils_fn2(x):
    return x**2


class TestUtils(ut.TestCase):
    def test_pickle_fn(self):
        fn1 = lambda x: x + 1
        fn3 = np.log10
        fn2 = utils_fn2
        hex1 = commons.fn_to_hex(fn1)
        hex2 = commons.fn_to_hex(fn2)
        hex3 = commons.fn_to_hex(np.log10)  # we are testing without var assign
        inp = 10
        r_fn1 = commons.hex_to_fn(hex1)
        r_fn2 = commons.hex_to_fn(hex2)
        r_fn3 = commons.hex_to_fn(hex3)
        self.assertListEqual([fn1(inp), fn2(inp), fn3(inp)], [r_fn1(inp), r_fn2(inp), r_fn3(inp)])
        with self.assertRaises(AttributeError):
            # we cannot pickle local function
            # this is impossible by design
            def fn2(x):
                return x**2

            hex2 = commons.fn_to_hex(fn2)

    def test_dtype(self):
        self.assertTrue(datatype.is_dtype_tensor(torch.int))
        self.assertFalse(datatype.is_dtype_tensor(int))
        self.assertTrue(datatype.is_dtype_numpy(np.float32))

    def test_null(self):
        self.assertTrue(datatype.is_null(None))
        self.assertTrue(datatype.is_null([float("nan"), np.NAN]))
        self.assertFalse(datatype.is_null([float("nan"), 1.0]))

    def test_cast(self):
        arr1 = np.random.randn(5, 1)
        dict1 = dict(test=np.array([2, 3, 1]))
        torch_arr1 = datatype.cast(arr1, torch.float)
        self.assertTrue(np.allclose(arr1, torch_arr1.cpu().numpy()))
        arr2 = None
        self.assertIsNone(datatype.cast(arr2, int))
        self.assertListEqual(list(datatype.cast(dict1, list)["test"]), list(dict1["test"]))

    def test_one_hot(self):
        val1 = 1
        val2 = 10
        allowable_set = range(5)
        enc1 = commons.one_hot_encoding(val1, allowable_set)
        enc2 = commons.one_hot_encoding(val2, allowable_set, encode_unknown=True)
        self.assertListEqual(list(enc1), [0, 1, 0, 0, 0])
        self.assertListEqual(list(enc2), [0, 0, 0, 0, 0, 1])


class TestCache(ut.TestCase):
    def test_datacache(self):
        mol_data = dm.data.freesolv()
        # in memory cache
        smiles_list = mol_data["smiles"].values
        cache = DataCache(name="test", delete_on_exit=True)
        featurizer = FPVecTransformer(kind="rdkit", length=10)
        expected_output = datatype.to_numpy(featurizer.transform(smiles_list))
        computed_data = datatype.to_numpy(cache(smiles_list, featurizer))
        np.testing.assert_array_equal(expected_output, computed_data)

        # refetch data from cache
        refetched_data = cache.fetch(smiles_list)
        refetched_data = datatype.to_numpy(refetched_data)
        np.testing.assert_array_equal(expected_output, refetched_data)

        # test cache on local storage with shelve
        disk_cache = DataCache(name="test2", cache_file=True, delete_on_exit=True)
        self.assertIsInstance(disk_cache.cache, shelve.Shelf)
        computed_data2 = datatype.to_numpy(disk_cache(smiles_list, featurizer))
        np.testing.assert_array_equal(expected_output, computed_data2)

        # check data retrieval
        first_smiles_val = expected_output[0]
        first_smiles = smiles_list[0]
        first_mol = dm.to_mol(first_smiles)
        # check data
        self.assertTrue(first_smiles in disk_cache)
        self.assertFalse("FAKE" in disk_cache)

        np.testing.assert_array_equal(disk_cache[first_smiles], first_smiles_val)
        np.testing.assert_array_equal(disk_cache[first_mol], first_smiles_val)

        # test saving and reloading cache
        save_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        save_file = save_file.name
        disk_cache.save_to_file(save_file)
        new_cache = DataCache.load_from_file(save_file)
        self.assertTrue(first_smiles in new_cache)
        np.testing.assert_array_equal(new_cache[first_smiles], first_smiles_val)
        try:
            os.unlink(save_file)
        except:
            pass

    def test_filecache(self):
        mol_data = dm.data.freesolv().iloc[:100]
        # in memory cache
        smiles_list = mol_data["smiles"].values
        hash_fn = dm.unique_id
        featurizer = FPVecTransformer(kind="rdkit", length=10)
        vals = datatype.to_numpy(featurizer.transform(smiles_list))
        mol_ids = [hash_fn(x) for x in smiles_list]

        first_smiles_val = vals[0]
        first_smiles = smiles_list[0]
        first_mol = dm.to_mol(first_smiles)

        precomputed_data = dict(zip(mol_ids, vals))
        # TEST USING A DATAFRAME
        df = pd.DataFrame(precomputed_data.items(), columns=["keys", "feats"])
        df["values"] = df["feats"].apply(commons.pack_bits)
        with tempfile.NamedTemporaryFile(delete=True, suffix="csv") as temp_file:
            df.to_csv(temp_file, index=False)
            cache = FileCache(temp_file.name, mol_hasher=hash_fn, file_type="csv")
            # check data
            self.assertTrue(first_smiles in cache)
            self.assertFalse("FAKE" in cache)
            np.testing.assert_array_equal(cache[first_smiles], first_smiles_val)

        with tempfile.NamedTemporaryFile(delete=True, suffix=".pkl") as temp_file:
            joblib.dump(precomputed_data, temp_file.name)
            cache_pkl = FileCache(temp_file.name, mol_hasher=hash_fn, file_type="pickle")
            # check data
            self.assertTrue(first_mol in cache_pkl)
            self.assertFalse("FAKE" in cache_pkl)
            np.testing.assert_array_equal(cache_pkl[first_smiles], first_smiles_val)

        with tempfile.NamedTemporaryFile(delete=True, suffix=".parquet") as temp_file:
            df_parquet = pd.DataFrame(precomputed_data.items(), columns=["keys", "feats"])
            df_parquet["values"] = df_parquet["feats"]
            df_parquet.to_parquet(temp_file.name)
            cache = FileCache(temp_file.name, mol_hasher=hash_fn, file_type="parquet")
            # check data
            self.assertTrue(first_mol in cache)
            self.assertFalse("FAKE" in cache)
            np.testing.assert_array_equal(cache[first_smiles], first_smiles_val)

        with tempfile.NamedTemporaryFile(delete=True, suffix=".h5") as temp_file:
            with h5py.File(temp_file, "w") as f:
                for k, v in precomputed_data.items():
                    f.create_dataset(k, data=v)

            cache = FileCache(temp_file.name, mol_hasher=hash_fn, file_type="hdf5")
            # check data
            self.assertTrue(first_smiles in cache)
            self.assertFalse("FAKE" in cache)
            self.assertEqual(len(cache), len(smiles_list))
            np.testing.assert_array_equal(cache[first_smiles], first_smiles_val)

            # refetch data from cache
            refetched_data = cache.fetch(smiles_list)
            refetched_data = datatype.to_numpy(refetched_data)
            np.testing.assert_array_equal(vals, refetched_data)

        # check cache updating and reloading reloading of cache
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pkl") as temp_file:
            smiles_list2 = dm.data.freesolv()["smiles"].iloc[50:150].values
            _ = cache(smiles_list2, featurizer)
            self.assertEqual(len(cache), 150)
            cache.save_to_file(temp_file.name, file_type="pickle")
            parquet_out = temp_file.name + ".parquet"
            csv_out = temp_file.name + ".csv"
            cache.save_to_file(parquet_out, file_type="parquet")
            cache.save_to_file(csv_out, file_type="csv")
            reloaded_cache = FileCache.load_from_file(temp_file.name, file_type="pickle")
            reloaded_cache_parquet = FileCache.load_from_file(parquet_out, file_type="parquet")
            reloaded_cache_csv = FileCache.load_from_file(csv_out, file_type="csv")
            self.assertTrue(first_smiles in reloaded_cache, msg=reloaded_cache)
            self.assertFalse("FAKE" in reloaded_cache)
            np.testing.assert_array_equal(reloaded_cache[first_smiles], first_smiles_val)
            np.testing.assert_array_equal(reloaded_cache_parquet[first_smiles], first_smiles_val)
            np.testing.assert_array_equal(reloaded_cache_csv[first_smiles], first_smiles_val)
            for path in [parquet_out, csv_out]:
                try:
                    os.unlink(path)
                except:
                    shutil.rmtree(path)

    def test_cache_list(self):
        # Test multiple cache simultaneously
        mol_data = dm.data.freesolv().iloc[:200]
        smiles_list = mol_data["smiles"].values
        featurizer = FPVecTransformer(kind="rdkit", length=10)
        vals = datatype.to_numpy(featurizer.transform(smiles_list))

        with tempfile.NamedTemporaryFile(delete=True, suffix="csv") as temp_file:
            cache1 = FileCache(temp_file.name, file_type="csv")
            cache1(smiles_list[:50], featurizer)

            cache2 = DataCache(name="test2", cache_file=True, delete_on_exit=True)
            cache2(smiles_list[50:100], featurizer)

            cache3 = DataCache(name="test2", cache_file=False, delete_on_exit=True)
            cache3(smiles_list[100:150], featurizer)

            cache_merge = CacheList(cache1, cache2)
            np.testing.assert_array_equal(cache_merge.fetch(smiles_list)[:100], vals[:100])

            cache_merge.update(cache3)
            np.testing.assert_array_equal(cache_merge.fetch(smiles_list)[:150], vals[:150])

            cache_merge.clear()
            self.assertTrue(len(cache1) == 0)
            self.assertTrue(len(cache2) == 0)
            # this one should not change
            self.assertTrue(len(cache3) != 0)

    def test_align_conformers(self):
        # Get some molecules
        smiles_list = [
            "Nc1cnn(-c2ccccc2)c(=O)c1Cl",
            "Cc1ccn(-c2ccccc2)c(=O)c1F",
            "Cc1cnn(-c2ccccc2)c(=O)c1Cl",
            "Cc1cnn(-c2ccccc2)c(=O)c1",
        ]
        mols = [dm.to_mol(smiles) for smiles in smiles_list]

        # Generate conformers
        mols = [dm.conformers.generate(mol) for mol in mols]

        # Save average coordinates (just for the assert below)
        dummy_pos_averages = [dm.conformers.get_coords(mol).mean() for mol in mols]

        # Align conformers
        mols, scores = commons.align_conformers(mols)

        # Get the average coordinates after aligning
        dummy_pos_averages_aligned = [dm.conformers.get_coords(mol).mean() for mol in mols]

        # Reference mol should have the same coordinate
        assert np.allclose(dummy_pos_averages[:1], dummy_pos_averages_aligned[:1])

        # Other molecules should have different coordinates
        for p1, p2 in zip(dummy_pos_averages[1:], dummy_pos_averages_aligned[1:]):
            assert p1 != p2
