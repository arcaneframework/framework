// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Hdf5Utils.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Utility functions for hdf5.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_HDF5_HDF5UTILS_H
#define ARCANE_HDF5_HDF5UTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/NumericTypes.h"

#include "arcane/datatype/DataTypes.h"

#include "arcane/hdf5/ArcaneHdf5Global.h"

// This macro for MSVC with DLLs, to avoid undefined external symbols
// undefined with H5T_NATIVE*
#define _HDF5USEDLL_
#include <hdf5.h>

#include <mutex>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// At least hdf5 1.8 is required
#if (H5_VERS_MAJOR < 2) && (H5_VERS_MAJOR == 1 && H5_VERS_MINOR < 10)
#error "This version of HDF5 is too old. Version 1.10+ is required"
#endif

// Keep these macros for compatibility but they will need to be removed.
#define ARCANE_HDF5_1_6_AND_AFTER
#define ARCANE_HDF5_1_8_AND_AFTER

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IParallelMng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Utility functions for Hdf5.
 */
namespace Arcane::Hdf5Utils
{
extern "C" {
ARCANE_HDF5_EXPORT herr_t _ArcaneHdf5UtilsGroupIterateMe(hid_t, const char*, void*);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_HDF5_EXPORT Hdf5Mutex
{

 public:

  Hdf5Mutex(std::mutex& mutex, bool& is_active)
  : m_mutex(mutex)
  , m_is_active(is_active)
  {}

 public:

  void lock() const
  {
    if (m_is_active)
      m_mutex.lock();
  }
  void unlock() const
  {
    if (m_is_active)
      m_mutex.unlock();
  }

 private:

  std::mutex& m_mutex;
  bool& m_is_active;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_HDF5_EXPORT Hdf5Mutex&
_ArcaneHdf5UtilsMutex();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Class serving as an initializer for HDF.
 *
 * This object allows safe initialization of HDF5 in multi-thread mode.
 */
class ARCANE_HDF5_EXPORT HInit
{
 public:

  HInit();

 public:

  //! True HDF5 is compiled with MPI support
  static constexpr bool hasParallelHdf5()
  {
#ifdef H5_HAVE_PARALLEL
    return true;
#else
    return false;
#endif
  }

  /*!
   * \brief Function allowing activation or deactivation of locks
   * on each HDF5 call.
   * \warning The environment variable ARCANE_HDF5_DISABLE_MUTEX is
   * prioritized over the parameter of this function.
   * \warning In hydride, if a hybrid parallelMng is used in parallel
   * and a full MPI parallelMng is used, and useMutex() is changed regularly,
   * be careful not to mix HDF5 calls with the
   * two parallelMngs.
   *
   * \param is_active true if mutexes are activated.
   */
  static void useMutex(bool is_active, IParallelMng* pm);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulates a hid_t.
 *
 * This class is not copyable.
 */
class ARCANE_HDF5_EXPORT Hid
{
 public:

  Hid() = default;
  Hid(hid_t id)
  : m_id(id)
  {}
  virtual ~Hid() {}

 protected:

  // This copy constructor will eventually need to be forbidden
  Hid(const Hid& hid)
  : m_id(hid.id())
  {}
  void _setId(hid_t id) { m_id = id; }
  void _setNullId() { m_id = -1; }

 private:

  Hid& operator=(const Hid& hid) = delete;

 public:

  hid_t id() const { return m_id; }
  bool isBad() const { return m_id < 0; }

 private:

  hid_t m_id = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulates a hid_t for a property (H5P*).
 */
class ARCANE_HDF5_EXPORT HProperty
: public Hid
{
 public:

  HProperty() { _setId(H5P_DEFAULT); }
  ~HProperty()
  {
    close();
  }
  HProperty(HProperty&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  HProperty& operator=(HProperty&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }

 public:

  HProperty(const HProperty& v) = delete;
  HProperty& operator=(const HProperty& hid) = delete;

 public:

  void close();

  void create(hid_t cls_id);
  void setId(hid_t new_id)
  {
    _setId(new_id);
  }

  /*!
   * \brief Creates a file property for MPIIO.
   *
   * Only works if HDF5 is compiled with MPI. Otherwise, it throws
   * an exception. If \a mpi_comm is the MPI communicator associated
   * with \a pm, calling this method creates a property as follows:
   *
   * \code
   * create(H5P_FILE_ACCESS);
   * H5Pset_fapl_mpio(id(), mpi_comm, MPI_INFO_NULL);
   * \endcode
   */
  void createFilePropertyMPIIO(IParallelMng* pm);

  /*!
   * \brief Creates a dataset property for MPIIO.
   *
   * Only works if HDF5 is compiled with MPI. Otherwise, it throws
   * an exception. Calling this method creates a property as follows:
   *
   * \code
   * create(H5P_DATASET_XFER);
   * H5Pset_dxpl_mpio(id(), H5FD_MPIO_COLLECTIVE);
   * H5Pset_selection_io(id(), H5D_SELECTION_IO_MODE_OFF);
   * \endcode
   */
  void createDatasetTransfertCollectiveMPIIO();

  /*!
   * \brief Creates a dataset property for MPIIO.
   *
   * Only works if HDF5 is compiled with MPI. Otherwise, it throws
   * an exception. Calling this method creates a property as follows:
   *
   * \code
   * create(H5P_DATASET_XFER);
   * H5Pset_dxpl_mpio(id(), H5FD_MPIO_INDEPENDENT);
   * H5Pset_selection_io(id(), H5D_SELECTION_IO_MODE_OFF);
   * \endcode
   */
  void createDatasetTransfertIndependentMPIIO();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulates a hid_t for a file.
 */
class ARCANE_HDF5_EXPORT HFile
: public Hid
{
 public:

  HFile() = default;
  ~HFile() { _close(); }
  HFile(HFile&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  HFile& operator=(HFile&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }
  HFile& operator=(const HFile& hid) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  HFile(const HFile& rhs)
  : Hid(rhs)
  {}

 public:

  void openTruncate(const String& var);
  void openAppend(const String& var);
  void openRead(const String& var);
  void openTruncate(const String& var, hid_t plist_id);
  void openAppend(const String& var, hid_t plist_id);
  void openRead(const String& var, hid_t plist_id);
  void close();

 private:

  herr_t _close();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Helper class for searching a group.
 */
class ARCANE_HDF5_EXPORT HGroupSearch
{
 public:

  HGroupSearch(const String& group_name)
  : m_group_name(group_name)
  {
  }

 public:

  herr_t iterateMe(const char* member_name)
  {
    //cerr << "** ITERATE <" << member_name << ">\n";
    if (m_group_name == member_name)
      return 1;
    return 0;
  }

 private:

  String m_group_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulates a hid_t for a group.
 */
class ARCANE_HDF5_EXPORT HGroup
: public Hid
{
 public:

  HGroup() {}
  ~HGroup() { close(); }
  HGroup(HGroup&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  HGroup& operator=(HGroup&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }
  HGroup& operator=(const HGroup& hid) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  HGroup(const HGroup& rhs)
  : Hid(rhs)
  {}

 public:

  void create(const Hid& loc_id, const String& group_name);
  void openOrCreate(const Hid& loc_id, const String& group_name);
  void recursiveCreate(const Hid& loc_id, const String& var);
  void recursiveCreate(const Hid& loc_id, const Array<String>& paths);
  void checkDelete(const Hid& loc_id, const String& var);
  void recursiveOpen(const Hid& loc_id, const String& var);
  void open(const Hid& loc_id, const String& var);
  void openIfExists(const Hid& loc_id, const Array<String>& var);
  bool hasChildren(const String& var);
  void close();
  static bool hasChildren(hid_t loc_id, const String& var);

 private:

  hid_t _checkOrCreate(hid_t loc_id, const String& group_name);
  hid_t _checkExist(hid_t loc_id, const String& group_name);

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulates a hid_t for a dataspace.
 */
class ARCANE_HDF5_EXPORT HSpace
: public Hid
{
 public:

  HSpace() {}
  explicit HSpace(hid_t id)
  : Hid(id)
  {}
  HSpace(HSpace&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  ~HSpace();
  HSpace& operator=(HSpace&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }
  HSpace& operator=(const HSpace& hid) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  HSpace(const HSpace& v)
  : Hid(v)
  {}

 public:

  void createSimple(int nb, hsize_t dims[]);
  void createSimple(int nb, hsize_t dims[], hsize_t max_dims[]);
  int nbDimension();
  herr_t getDimensions(hsize_t dims[], hsize_t max_dims[]);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulates a hid_t for a dataset.
 */
class ARCANE_HDF5_EXPORT HDataset
: public Hid
{
 public:

  HDataset() {}
  ~HDataset() { close(); }
  HDataset(HDataset&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  HDataset& operator=(HDataset&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }
  HDataset& operator=(const HDataset& hid) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  HDataset(const HDataset& v)
  : Hid(v)
  {}

 public:

  void close();
  void create(const Hid& loc_id, const String& var, hid_t save_type, const HSpace& space_id, hid_t plist);
  void create(const Hid& loc_id, const String& var, hid_t save_type,
              const HSpace& space_id, const HProperty& link_plist,
              const HProperty& creation_plist, const HProperty& access_plist);
  void recursiveCreate(const Hid& loc_id, const String& var, hid_t save_type, const HSpace& space_id, hid_t plist);
  void open(const Hid& loc_id, const String& var);
  void openIfExists(const Hid& loc_id, const String& var);
  herr_t write(hid_t native_type, const void* array);
  herr_t write(hid_t native_type, const void* array, const HSpace& memspace_id,
               const HSpace& filespace_id, hid_t plist);
  herr_t write(hid_t native_type, const void* array, const HSpace& memspace_id,
               const HSpace& filespace_id, const HProperty& plist);
  herr_t read(hid_t native_type, void* array);
  void readWithException(hid_t native_type, void* array);
  HSpace getSpace();
  herr_t setExtent(const hsize_t new_dims[]);

 private:

  void _remove(hid_t hid, const String& var);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulates a hid_t for an attribute.
 */
class ARCANE_HDF5_EXPORT HAttribute
: public Hid
{
 public:

  HAttribute() {}
  ~HAttribute();
  HAttribute(HAttribute&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  HAttribute& operator=(HAttribute&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }
  HAttribute& operator=(const HAttribute& hid) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  HAttribute(const HAttribute& v)
  : Hid(v)
  {}

 public:

  void remove(const Hid& loc_id, const String& var);
  void create(const Hid& loc_id, const String& var, hid_t save_type, const HSpace& space_id);
  void open(const Hid& loc_id, const String& var);
  herr_t write(hid_t native_type, void* array);
  herr_t read(hid_t native_type, void* array);
  HSpace getSpace();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulates a hid_t for a type.
 */
class ARCANE_HDF5_EXPORT HType
: public Hid
{
 public:

  HType() {}
  ~HType();
  HType(HType&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  HType& operator=(HType&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }
  HType& operator=(const HType& hid) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  HType(const HType& v)
  : Hid(v)
  {}

 public:

  void setId(hid_t new_id)
  {
    _setId(new_id);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Definition of standard Arcane types for hdf5.
 *
 * An instance of this class constructs HDF5 types to perform the
 * conversion between HDF5 types and Arcane types.
 *
 * The default constructor using HDF5 calls is not thread-safe.
 * If running in a multi-threaded context, it is preferable to use
 * StandardTypes(false) and call init() to initialize the types.
 */
class ARCANE_HDF5_EXPORT StandardTypes
{
 public:

  /*!
   * \brief Creates an instance by initializing the types.
   *
   * \warning not thread-safe.
   */
  StandardTypes();

  //! Creates an instance without initializing the types, i.e., do_init is false.
  explicit StandardTypes(bool do_init);

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  StandardTypes(const StandardTypes& rhs) = default;

  ~StandardTypes();

  StandardTypes& operator=(const StandardTypes& rhs) = delete;

 public:

  //! Initializes the types.
  void initialize();

 public:

  hid_t nativeType(float) const { return H5T_NATIVE_FLOAT; }
  hid_t nativeType(double) const { return H5T_NATIVE_DOUBLE; }
  hid_t nativeType(Real2) const { return m_real2_id.id(); }
  hid_t nativeType(Real3) const { return m_real3_id.id(); }
  hid_t nativeType(Real2x2) const { return m_real2x2_id.id(); }
  hid_t nativeType(Real3x3) const { return m_real3x3_id.id(); }
  hid_t nativeType(long double) const { return H5T_NATIVE_LDOUBLE; }
  hid_t nativeType(unsigned int) const { return H5T_NATIVE_UINT; }
  hid_t nativeType(unsigned long) const { return H5T_NATIVE_ULONG; }
  hid_t nativeType(unsigned long long) const { return H5T_NATIVE_ULLONG; }
  hid_t nativeType(int) const { return H5T_NATIVE_INT; }
  hid_t nativeType(long long) const { return H5T_NATIVE_LLONG; }
  hid_t nativeType(long) const { return H5T_NATIVE_LONG; }
  hid_t nativeType(char) const { return H5T_NATIVE_CHAR; }
  hid_t nativeType(unsigned char) const { return H5T_NATIVE_UCHAR; }
  hid_t nativeType(signed char) const { return H5T_NATIVE_SCHAR; }
  hid_t nativeType(unsigned short) const { return H5T_NATIVE_USHORT; }
  hid_t nativeType(short) const { return H5T_NATIVE_SHORT; }
#ifdef ARCANE_REAL_NOT_BUILTIN
  hid_t nativeType(Real) const;
#endif
  hid_t nativeType(eDataType sd) const;
  hid_t nativeType(BFloat16) const { return m_bfloat16_id.id(); }
  hid_t nativeType(Float16) const { return m_float16_id.id(); }

 public:

  hid_t saveType(float) const
  {
    return m_float32_id.id();
  }
  hid_t saveType(double) const
  {
    return m_real_id.id();
  }
  hid_t saveType(Real2) const
  {
    return m_real2_id.id();
  }
  hid_t saveType(Real3) const
  {
    return m_real3_id.id();
  }
  hid_t saveType(Real2x2) const
  {
    return m_real2x2_id.id();
  }
  hid_t saveType(Real3x3) const
  {
    return m_real3x3_id.id();
  }
  hid_t saveType(long double) const
  {
    return m_real_id.id();
  }
  hid_t saveType(short) const
  {
    return m_short_id.id();
  }
  hid_t saveType(unsigned short) const
  {
    return m_ushort_id.id();
  }
  hid_t saveType(unsigned int) const
  {
    return m_uint_id.id();
  }
  hid_t saveType(unsigned long) const
  {
    return m_ulong_id.id();
  }
  hid_t saveType(unsigned long long) const
  {
    return m_ulong_id.id();
  }
  hid_t saveType(int) const
  {
    return m_int_id.id();
  }
  hid_t saveType(long) const
  {
    return m_long_id.id();
  }
  hid_t saveType(long long) const
  {
    return m_long_id.id();
  }
  hid_t saveType(char) const
  {
    return m_char_id.id();
  }
  hid_t saveType(unsigned char) const
  {
    return m_uchar_id.id();
  }
  hid_t saveType(signed char) const
  {
    return m_schar_id.id();
  }
  hid_t saveType(BFloat16) const
  {
    return m_bfloat16_id.id();
  }
  hid_t saveType(Float16) const
  {
    return m_float16_id.id();
  }
#ifdef ARCANE_REAL_NOT_BUILTIN
  hid_t saveType(Real) const
  {
    return m_real_id.id();
  }
#endif
  hid_t saveType(eDataType sd) const;

 private:

  /*!
   * \brief Class initializing HDF.
   *
   * \warning This instance must always be defined before members that
   * use HDF5 so that initialization happens first and resource release
   * happens last.
   */
  HInit m_init;

 public:

  HType m_char_id; //!< HDF identifier for characters
  HType m_uchar_id; //!< HDF identifier for unsigned characters
  HType m_schar_id; //!< HDF identifier for signed characters
  HType m_short_id; //!< HDF identifier for signed shorts
  HType m_ushort_id; //!< HDF identifier for unsigned shorts
  HType m_int_id; //!< HDF identifier for signed integers
  HType m_long_id; //!< HDF identifier for signed longs
  HType m_uint_id; //!< HDF identifier for unsigned integers
  HType m_ulong_id; //!< HDF identifier for unsigned longs
  HType m_real_id; //!< HDF identifier for reals
  HType m_real2_id; //!< HDF identifier for Real2
  HType m_real3_id; //!< HDF identifier for Real3
  HType m_real2x2_id; //!< HDF identifier for Real2x2
  HType m_real3x3_id; //!< HDF identifier for Real3x3
  HType m_float16_id; //!< HDF identifier for Float16
  HType m_bfloat16_id; //!< HDF identifier for BFloat16
  HType m_float32_id; //!< HDF identifier for Float32

 private:

  void _H5Tinsert(hid_t type, const char* name, Integer offset, hid_t field_id);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulates a simple dataset from an HDF5 file that represents
 * an array.
 */
class ARCANE_HDF5_EXPORT StandardArray
{
 public:

  StandardArray(hid_t hfile, const String& hpath);
  virtual ~StandardArray() {}

 public:

  /*!
   * \brief When reading, positions the path in \a hfile to the dataset containing the unique_ids.
   *
   * This call is optional but if used, it must be done before
   * reading the values.
   */
  void setIdsPath(const String& ids_path);
  void readDim();
  Int64ConstArrayView dimensions() const { return m_dimensions; }
  virtual bool exists() const;

 protected:

  void _write(const void* buffer, Integer nb_element, hid_t save_type, hid_t native_type);

 protected:

  hid_t m_hfile;
  String m_hpath;
  String m_ids_hpath;
  HDataset m_hdataset;
  HDataset m_ids_dataset;
  Int64UniqueArray m_dimensions;
  bool m_is_init;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulates a simple dataset from an HDF5 file that represents
 * an array.
 */
template <typename DataType>
class ARCANE_HDF5_EXPORT StandardArrayT
: public StandardArray
{
 private:

  struct ValueWithUid
  {
   public:

    Int64 m_uid;
    Integer m_index;

   public:

    bool operator<(const ValueWithUid& rhs) const
    {
      return m_uid < rhs.m_uid;
    }
  };

 public:

  StandardArrayT(hid_t hfile, const String& hpath);

 public:

  /*!
   * \brief Reads the dataset of a 1D array.
   * This operation is only valid after calling readDim().
   * \a buffer must have been allocated.
   * To read directly, use directRead()
   */
  void read(StandardTypes& st, ArrayView<DataType> buffer);
  /*!
   * \brief Reads the dataset of a 1D array.
   */
  void directRead(StandardTypes& st, Array<DataType>& buffer);
  void parallelRead(IParallelMng* pm, StandardTypes& st,
                    Array<DataType>& buffer, Int64Array& unique_ids);
  void write(StandardTypes& st, ConstArrayView<DataType> buffer);
  void parallelWrite(IParallelMng* pm, StandardTypes& st,
                     ConstArrayView<DataType> buffer,
                     Int64ConstArrayView unique_ids);

 private:

  void _writeSortedValues(ITraceMng* tm, StandardTypes& st, ConstArrayView<DataType> buffer,
                          Int64ConstArrayView unique_ids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Encapsulates a simple dataset from an HDF5 file that represents a scalar (possibly String).
 */
template <typename DataType>
class ARCANE_HDF5_EXPORT StandardScalarT
{
 public:

  //! Constructor
  StandardScalarT(hid_t hfile, const String& hpath)
  : m_hfile(hfile)
  , m_hpath(hpath)
  {}

 public:

  //! Reads a data item
  DataType read(Hdf5Utils::StandardTypes& st);

  //! Writes a data item
  void write(Hdf5Utils::StandardTypes& st, const DataType& t);

 protected:

  hid_t m_hfile;
  String m_hpath;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Hdf5Utils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
