/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <alien/utils/Precomp.h>

namespace Alien
{
struct HDF5Base
{
  typedef enum
  {
    ASCII,
    HDF5,
    SMART
  } eFormatType;

  struct FileNode
  {
    FileNode(int _level = 0)
    : name("")
    , path_name("")
    , level(_level)
#ifdef ALIEN_USE_HDF5
    , h_id(-1)
#endif
#ifdef ALIEN_USE_LIBXML2
    , x_id(NULL)
#endif
    {}

    std::string name;
    std::string path_name;
    int level;

#ifdef ALIEN_USE_HDF5
    hid_t h_id;
#endif
#ifdef ALIEN_USE_LIBXML2
    xmlNodePtr x_id;
#endif
  };

  class StandardTypes
  {
   public:
    StandardTypes() {}
    ~StandardTypes() {}

   public:
#ifdef ALIEN_USE_HDF5
    hid_t nativeType(float) const
    {
      return H5T_NATIVE_FLOAT;
    }
    hid_t nativeType(double) const { return H5T_NATIVE_DOUBLE; }
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
// hid_t nativeType(eDataType sd) const;
#endif
    std::string type(double) const
    {
      return "double";
    }
    std::string type(float) const { return "float"; }
    std::string type(int) const { return "int32"; }
    std::string type(Int64) const { return "int64"; }

    std::string type(char) const { return "char"; }

   public:
#ifdef ALIEN_USE_HDF5
    hid_t m_char_id; //!< Identifiant HDF des entiers sign�s
    hid_t m_uchar_id; //!< Identifiant HDF des caract�res non-sign�s
    hid_t m_int_id; //!< Identifiant HDF des entiers sign�s
    hid_t m_long_id; //!< Identifiant HDF des entiers long sign�s
    hid_t m_uint_id; //!< Identifiant HDF des entiers non sign�s
    hid_t m_ulong_id; //!< Identifiant HDF des entiers long non sign�s
    hid_t m_real_id; //!< Identifiant HDF des r�els
#endif
  };

  HDF5Base(std::string const& name)
  : name(name)
  , xfile_name(name + ".xml")
  , hfile_name(name + ".h5")
  , format("ascii")
  , type(HDF5)
  {}

  std::string name;
  std::string xfile_name;
  std::string hfile_name;
  std::string format;
  eFormatType type;
  StandardTypes m_types;
#ifdef ALIEN_USE_LIBXML2
  xmlDocPtr doc;
#endif
#ifdef ALIEN_USE_HDF5
  hid_t hfile;
#endif
  std::vector<double> rbuffer;
  std::vector<Int64> i64buffer;
  std::vector<int> i32buffer;
};

struct Exporter : public HDF5Base
{
  Exporter(std::string const& name, std::string const& out_format, int prec,
           int smart_size_limit = 4)
  : HDF5Base(name)
  , m_smart_size_limit(smart_size_limit)
  , m_write_xml_hdf(false)
  {
    if (out_format.compare("ascii") == 0) {
      format = "xml";
      type = ASCII;
      fout.open(xfile_name.c_str());
      // fout<<std::fixed<<std::setprecision(prec) ;
      fout << "<?xml version=\"1.0\" encoding=\"iso-8859-1\" ?>" << std::endl;
    }
    if (out_format.compare("hdf5") == 0) {
      format = "hdf";
      type = HDF5;
      if (m_write_xml_hdf) {
        fout.open(xfile_name.c_str());
        // fout<<std::fixed<<std::setprecision(prec) ;
        fout << "<?xml version=\"1.0\" encoding=\"iso-8859-1\" ?>" << std::endl;
      }
#ifdef ALIEN_USE_HDF5
      hfile = H5Fcreate(hfile_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
#else
      throw FatalErrorException(A_FUNCINFO,
                                "hdf5 format requested while there is no support for hdf5 in Alien");
#endif
    }
    if (out_format.compare("smart") == 0) {
      format = "smart";
      type = SMART;
      fout.open(xfile_name.c_str());
      // fout<<std::fixed<<std::setprecision(prec) ;
      fout << "<?xml version=\"1.0\" encoding=\"iso-8859-1\" ?>" << std::endl;
#ifdef ALIEN_USE_HDF5
      hfile = H5Fcreate(hfile_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
#else
      throw FatalErrorException(A_FUNCINFO,
                                "smart format requested while there is no support for hdf5 in Alien");
#endif
    }
  }
  ~Exporter()
  {
    if (type == ASCII || m_write_xml_hdf || type == SMART) {
      fout.close();
    }
    if (type == HDF5 || type == SMART) {
#ifdef ALIEN_USE_HDF5
      H5Fclose(hfile);
#endif
    }
  }

  // create a node at base (without parent)
  FileNode createFileNode(const std::string& name, int level = 0)
  {
    FileNode node;
    node.name = name;
    node.path_name = "/" + name;
    node.level = level;
    if (type == ASCII || m_write_xml_hdf || type == SMART) {
      fout << "<" << name << ">" << std::endl;
    }
    if (type == HDF5 || type == SMART) {
#ifdef ALIEN_USE_HDF5
      node.h_id =
      H5Gcreate2(hfile, node.name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
    }
    return node;
  }

  FileNode createFileNode(FileNode const& parent, const std::string& name)
  {
    FileNode node;
    node.name = name;
    node.path_name = parent.path_name + "/" + name;
    node.level = parent.level + 1;
    if (type == ASCII || m_write_xml_hdf || type == SMART) {
      _tab(parent.level);
      fout << "<" << name << ">" << std::endl;
    }
    if (type == HDF5 || type == SMART) {
#ifdef ALIEN_USE_HDF5
      node.h_id = H5Gcreate2(
      parent.h_id, node.name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
    }
    return node;
  }

  FileNode createFileNode(
  FileNode const& parent, std::string name, std::string group_name)
  {
    FileNode node;
    node.name = name;
    node.path_name = parent.path_name + "/" + name;
    node.level = parent.level + 1;
    if (type == ASCII || m_write_xml_hdf || type == SMART) {
      for (int i = 0; i < parent.level; ++i)
        fout << "\t";
      fout << "<" << name << " group-name=\"" << group_name << "\">" << std::endl;
    }
    if (type == HDF5 || type == SMART) {
#ifdef ALIEN_USE_HDF5
      node.h_id = H5Gcreate2(
      parent.h_id, node.name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
    }
    return node;
  }

  FileNode createFileNode(FileNode const& parent, std::string name, std::string att_name,
                          std::string att_kind, bool is_group = false)
  {
    FileNode node;
    node.name = name;
    if (is_group)
      node.path_name = parent.path_name + "/" + name + "_" + att_kind + "_" + att_name;
    else
      node.path_name = parent.path_name + "/" + att_name;

    node.level = parent.level + 1;
    if (type == ASCII || m_write_xml_hdf || type == SMART) {
      _tab(parent.level);
      fout << "<" << name << " name=\"" << att_name << "\" kind=\"" << att_kind << "\">"
           << std::endl;
    }
    if (type == HDF5 || type == SMART) {
#ifdef ALIEN_USE_HDF5
      node.h_id = H5Gcreate2(
      parent.h_id, att_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
    }
    return node;
  }

  void closeFileNode(FileNode const& group)
  {
    if (type == ASCII || m_write_xml_hdf || type == SMART) {
      _tab(group.level - 1);
      fout << "</" << group.name << ">" << std::endl;
    }
    if (type == HDF5 || type == SMART) {
#ifdef ALIEN_USE_HDF5
      H5Gclose(group.h_id);
#endif
    }
  }

  template <typename ValueT>
  void write(FileNode const& parent_node, const std::string& node_name, const ValueT& val)
  {
    switch (type) {
    case ASCII: {
      _tab(parent_node.level);
      fout << "<" << node_name << " format=\"xml\"  type=\"" << m_types.type(ValueT())
           << "\">" << std::endl;
      fout << val << std::endl;
      _tab(parent_node.level);
      fout << "</" << node_name << ">" << std::endl;
    } break;
    case HDF5: {
      if (m_write_xml_hdf) {
        _tab(parent_node.level);
        fout << "<" << node_name << " format=\"hdf\"  type=\"" << m_types.type(ValueT())
             << "\">" << std::endl;
        _tab(parent_node.level);
        fout << hfile_name << ":" << parent_node.path_name << "/" << node_name
             << std::endl;
        _tab(parent_node.level);
        fout << "</" << node_name << ">" << std::endl;
      }
#ifdef ALIEN_USE_HDF5
      hsize_t dim = 1;
      hid_t dataspace_id = H5Screate_simple(1, &dim, NULL);
      hid_t dataset_id =
      H5Dcreate2(parent_node.h_id, node_name.c_str(), m_types.nativeType(ValueT()),
                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      herr_t status = H5Dwrite(
      dataset_id, m_types.nativeType(ValueT()), H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);
      if (status)
        std::cerr << "Error while writing HDF5 data set" << std::endl;
      H5Dclose(dataset_id);
      H5Sclose(dataspace_id);
#endif
    } break;
    case SMART: {
      _tab(parent_node.level);
      fout << "<" << node_name << " format=\"xml\"  type=\"" << m_types.type(ValueT())
           << "\">" << std::endl;
      fout << val << std::endl;
      _tab(parent_node.level);
      fout << "</" << node_name << ">" << std::endl;
#ifdef ALIEN_USE_HDF5
      hsize_t dim = 1;
      hid_t dataspace_id = H5Screate_simple(1, &dim, NULL);
      hid_t dataset_id =
      H5Dcreate2(parent_node.h_id, node_name.c_str(), m_types.nativeType(ValueT()),
                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      herr_t status = H5Dwrite(
      dataset_id, m_types.nativeType(ValueT()), H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);
      if (status)
        std::cerr << "Error while writing HDF5 data set" << std::endl;
      H5Dclose(dataset_id);
      H5Sclose(dataspace_id);
#endif
    } break;
    default:
      std::cerr << "Unknown output format " << std::endl;
      break;
    }
  }

  template <typename ValueT>
  void write(FileNode const& parent_node, const std::string& node_name,
             std::vector<ValueT>& buffer, int nb_elems_per_line = 1)
  {
    switch (type) {
    case ASCII: {
      _tab(parent_node.level);
      fout << "<" << node_name << " format=\"xml\"  type=\"" << m_types.type(ValueT())
           << "\">" << std::endl;

      int icount = 0;
      for (std::size_t n = 0; n < buffer.size() / nb_elems_per_line; ++n) {
        for (int k = 0; k < nb_elems_per_line; ++k)
          fout << buffer[icount + k] << " ";
        icount += nb_elems_per_line;
        fout << std::endl;
      }
      _tab(parent_node.level);
      fout << "</" << node_name << ">" << std::endl;
    } break;
    case HDF5: {
      if (m_write_xml_hdf) {
        _tab(parent_node.level);
        fout << "<" << node_name << " format=\"hdf\"  type=\"" << m_types.type(ValueT())
             << "\">" << std::endl;
        _tab(parent_node.level);
        fout << hfile_name << ":" << parent_node.path_name << "/" << node_name
             << std::endl;
        _tab(parent_node.level);
        fout << "</" << node_name << ">" << std::endl;
      }
#ifdef ALIEN_USE_HDF5
      hsize_t dim = buffer.size();
      hid_t dataspace_id = H5Screate_simple(1, &dim, NULL);
      hid_t dataset_id =
      H5Dcreate2(parent_node.h_id, node_name.c_str(), m_types.nativeType(ValueT()),
                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      herr_t status = H5Dwrite(dataset_id, m_types.nativeType(ValueT()), H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, buffer.data());
      if (status)
        std::cerr << "Error while writing HDF5 data set" << std::endl;
      H5Dclose(dataset_id);
      H5Sclose(dataspace_id);
#endif
    } break;
    case SMART: {
      if ((int)buffer.size() < m_smart_size_limit) {
        _tab(parent_node.level);
        fout << "<" << node_name << " format=\"xml\"  type=\"" << m_types.type(ValueT())
             << "\">" << std::endl;

        int icount = 0;
        for (std::size_t n = 0; n < buffer.size() / nb_elems_per_line; ++n) {
          for (int k = 0; k < nb_elems_per_line; ++k)
            fout << buffer[icount + k] << " ";
          icount += nb_elems_per_line;
          fout << std::endl;
        }
        _tab(parent_node.level);
        fout << "</" << node_name << ">" << std::endl;
      }
      else {
        _tab(parent_node.level);
        fout << "<" << node_name << " format=\"hdf\"  type=\"" << m_types.type(ValueT())
             << "\">" << std::endl;
        _tab(parent_node.level + 1);
        fout << hfile_name << ":" << parent_node.path_name << "/" << node_name
             << std::endl;
        _tab(parent_node.level);
        fout << "</" << node_name << ">" << std::endl;
      }
#ifdef ALIEN_USE_HDF5
      hsize_t dim = buffer.size();
      hid_t dataspace_id = H5Screate_simple(1, &dim, NULL);
      hid_t dataset_id =
      H5Dcreate2(parent_node.h_id, node_name.c_str(), m_types.nativeType(ValueT()),
                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      herr_t status = H5Dwrite(dataset_id, m_types.nativeType(ValueT()), H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, buffer.data());
      if (status)
        std::cerr << "Error while writing HDF5 data set" << std::endl;
      H5Dclose(dataset_id);
      H5Sclose(dataspace_id);
#endif
    } break;
    default:
      std::cerr << "Unknown output format " << std::endl;
      break;
    }
    buffer.clear();
  }

  void write(FileNode const& parent_node, const std::string& node_name,
             const std::string& buffer)
  {
    switch (type) {
    case ASCII: {
      _tab(parent_node.level);
      fout << "<" << node_name << " format=\"xml\"  type=\"" << m_types.type(char())
           << "\">" << std::endl;
      fout << buffer << std::endl;
      _tab(parent_node.level);
      fout << "</" << node_name << ">" << std::endl;
    } break;
    case HDF5: {
      if (m_write_xml_hdf) {
        _tab(parent_node.level);
        fout << "<" << node_name << " format=\"hdf\"  type=\"" << m_types.type(char())
             << "\">" << std::endl;
        _tab(parent_node.level);
        fout << hfile_name << ":" << parent_node.path_name << std::endl;
        _tab(parent_node.level);
        fout << "</" << node_name << ">" << std::endl;
      }
#ifdef ALIEN_USE_HDF5
      hsize_t dim = buffer.size();
      hid_t dataspace_id = H5Screate_simple(1, &dim, NULL);
      hid_t dataset_id =
      H5Dcreate2(parent_node.h_id, node_name.c_str(), m_types.nativeType(char()),
                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      herr_t status = H5Dwrite(dataset_id, m_types.nativeType(char()), H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, buffer.c_str());
      if (status)
        std::cerr << "Error while writing HDF5 data set" << std::endl;
      H5Dclose(dataset_id);
      H5Sclose(dataspace_id);
#endif
    } break;
    case SMART: {
      _tab(parent_node.level);
      fout << "<" << node_name << " format=\"xml\"  type=\"" << m_types.type(char())
           << "\">" << std::endl;
      fout << buffer << std::endl;
      _tab(parent_node.level);
      fout << "</" << node_name << ">" << std::endl;
#ifdef ALIEN_USE_HDF5
      hsize_t dim = buffer.size();
      hid_t dataspace_id = H5Screate_simple(1, &dim, NULL);
      hid_t dataset_id =
      H5Dcreate2(parent_node.h_id, node_name.c_str(), m_types.nativeType(char()),
                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      herr_t status = H5Dwrite(dataset_id, m_types.nativeType(char()), H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, buffer.data());
      if (status)
        std::cerr << "Error while writing HDF5 data set" << std::endl;
      H5Dclose(dataset_id);
      H5Sclose(dataspace_id);
#endif
    } break;
    default:
      std::cerr << "Unknown output format " << std::endl;
      break;
    }
  }

  std::ofstream fout;
  int m_smart_size_limit;
  bool m_write_xml_hdf;

  void _tab(const int level)
  {
    for (int i = 0; i < level; ++i)
      fout << "\t";
  }
};

struct Importer : public HDF5Base
{

  Importer(std::string const& name, std::string const& in_format, int prec)
  : HDF5Base(name)
  {
    if (in_format.compare("ascii") == 0) {
      format = "xml";
      type = ASCII;
#ifdef ALIEN_USE_LIBXML2
      doc = xmlParseFile(xfile_name.c_str());
      if (doc == NULL) {
        std::cerr << "Error while parsing XML file : " << xfile_name << std::endl;
      }
#else
      throw FatalErrorException(
      A_FUNCINFO, "xml format requested while there is no support for xml in Alien");
#endif
    }
    else if (in_format.compare("hdf5") == 0) {
      format = "hdf";
      type = HDF5;
#ifdef ALIEN_USE_HDF5
      hfile = H5Fopen(hfile_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
#else
      throw FatalErrorException(A_FUNCINFO,
                                "hdf5 format requested while there is no support for hdf5 in Alien");
#endif
    }
    else {
      throw FatalErrorException(
      A_FUNCINFO, "format requested not supported for read operations");
    }
  }

  ~Importer()
  {
    switch (type) {
    case ASCII:
#ifdef ALIEN_USE_LIBXML2
      xmlFreeDoc(doc);
#endif
      break;
    case HDF5: {
#ifdef ALIEN_USE_HDF5
      H5Fclose(hfile);
#endif
    } break;
    default:
      break;
    }
  }

  // open a file node with no parent
  FileNode openFileNode(const std::string& node_name)
  {
    FileNode node;

    switch (type) {
    case ASCII:
#ifdef ALIEN_USE_LIBXML2
      node.x_id = xmlDocGetRootElement(doc);
      if (xmlStrcmp(node.x_id->name, (const xmlChar*)node_name.c_str())) {
        throw FatalErrorException(A_FUNCINFO, "node not found");
      }
#endif
      break;
    case HDF5:
#ifdef ALIEN_USE_HDF5
      node.h_id = H5Gopen2(hfile, node.name.c_str(), H5P_DEFAULT);
#endif
      break;
    default:
      throw FatalErrorException(
      A_FUNCINFO, "format requested not supported for read operations");
      break;
    }
    return node;
  }

  FileNode openFileNode(FileNode const& parent, const std::string& name)
  {
    FileNode node;
    node.name = name;
    node.path_name = parent.path_name + "/" + name;
    node.level = parent.level + 1;
    switch (type) {
    case ASCII: {
#ifdef ALIEN_USE_LIBXML2
      xmlNodePtr cur = parent.x_id->xmlChildrenNode;
      while (cur != NULL) {
        if ((!xmlStrcmp(cur->name, (const xmlChar*)name.c_str()))) {
          node.x_id = cur;
          break;
        }
        cur = cur->next;
      }
// TODO: manage error when name not found
#endif
    } break;
    case HDF5: {
#ifdef ALIEN_USE_HDF5
      /* Open an existing dataset. */
      node.h_id = H5Gopen2(parent.h_id, node.name.c_str(), H5P_DEFAULT);
// TODO: manage error when name not found
#endif
    } break;
    default:
      throw FatalErrorException(
      A_FUNCINFO, "format requested not supported for read operations");
      break;
    }
    return node;
  }

  void closeFileNode(FileNode const& group)
  {
    switch (type) {
    case ASCII: {
      // xmlFree(group.x_id) ;
    } break;
    case HDF5: {
#ifdef ALIEN_USE_HDF5
      H5Gclose(group.h_id);
#endif
    } break;
    default:
      throw FatalErrorException(
      A_FUNCINFO, "format requested not supported for read operations");
      break;
    }
  }

  template <typename ValueT>
  void read(FileNode const& parent_node, const std::string& node_name,
            std::vector<ValueT>& buffer)
  {
    switch (type) {
    case ASCII: {
#ifdef ALIEN_USE_LIBXML2
      FileNode node;

      xmlNodePtr cur = parent_node.x_id->xmlChildrenNode;
      while (cur != nullptr) {
        if ((!xmlStrcmp(cur->name, (const xmlChar*)node_name.c_str()))) {
          node.x_id = cur;
          break;
        }
        cur = cur->next;
      }
      // TODO: handle when not found

      xmlChar* contenu = xmlNodeGetContent(node.x_id);
      // printf("%s\n",contenu);
      std::stringstream flux;
      flux << (const xmlChar*)contenu;
      for (std::size_t i = 0; i < buffer.size(); ++i)
        flux >> buffer[i];
      xmlFree(contenu);
#endif
    } break;
    case HDF5: {
#ifdef ALIEN_USE_HDF5
      /* Open an existing dataset. */
      hid_t dataset_id = H5Dopen2(parent_node.h_id, node_name.c_str(), H5P_DEFAULT);
      hid_t dataspace_id = H5Dget_space(dataset_id);

      int ndim = H5Sget_simple_extent_ndims(dataspace_id);

      if (ndim != 1) {
        throw FatalErrorException(A_FUNCINFO, "Error while reading HDF5 data set");
      }
      hsize_t size, maxsize;
      herr_t status = H5Sget_simple_extent_dims(dataspace_id, &size, &maxsize);
      if (status) {
        throw FatalErrorException(A_FUNCINFO, "Error while reading HDF5 data set");
      }
      if (size != buffer.size()) {
        // buffer size and data must match
        throw FatalErrorException(A_FUNCINFO, "Error while reading HDF5 data set");
      }

      status = H5Dread(dataset_id, m_types.nativeType(ValueT()), H5S_ALL, H5S_ALL,
                       H5P_DEFAULT, buffer.data());
      if (status)
        throw FatalErrorException(A_FUNCINFO, "Error while reading HDF5 node");

      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
#endif
    } break;
    default:
      throw FatalErrorException(A_FUNCINFO, "Unknown input format ");
      break;
    }
  }

  template <typename ValueT>
  void read(FileNode const& parent_node, const std::string& node_name, ValueT& val)
  {
    switch (type) {
    case ASCII: {
#ifdef ALIEN_USE_LIBXML2
      FileNode node;

      xmlNodePtr cur = parent_node.x_id->xmlChildrenNode;
      while (cur != nullptr) {
        if ((!xmlStrcmp(cur->name, (const xmlChar*)node_name.c_str()))) {
          node.x_id = cur;
          break;
        }
        cur = cur->next;
      }
      // TODO: handle when not found

      xmlChar* contenu = xmlNodeGetContent(node.x_id);
      // printf("%s\n",contenu);
      std::stringstream flux;
      flux << (const xmlChar*)contenu;
      flux >> val;
      xmlFree(contenu);
#endif
    } break;
    case HDF5: {
#ifdef ALIEN_USE_HDF5
      /* Open an existing dataset. */
      hid_t dataset_id = H5Dopen2(parent_node.h_id, node_name.c_str(), H5P_DEFAULT);
      hid_t dataspace_id = H5Dget_space(dataset_id);

      int ndim = H5Sget_simple_extent_ndims(dataspace_id);

      if (ndim != 1) {
        throw FatalErrorException(A_FUNCINFO, "Error while reading HDF5 data set");
      }
      hsize_t size, maxsize;
      herr_t status = H5Sget_simple_extent_dims(dataspace_id, &size, &maxsize);
      if (status) {
        throw FatalErrorException(A_FUNCINFO, "Error while reading HDF5 data set");
      }
      if (size != 1) {
        // buffer size and data must match
        throw FatalErrorException(A_FUNCINFO, "Error while reading HDF5 data set");
      }

      status = H5Dread(
      dataset_id, m_types.nativeType(ValueT()), H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);
      if (status)
        throw FatalErrorException(A_FUNCINFO, "Error while reading HDF5 node");

      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
#endif
    } break;
    default:
      throw FatalErrorException(A_FUNCINFO, "Unknown input format ");
      break;
    }
  }

  void read(FileNode const& node, std::string& buffer)
  {
    switch (type) {
    case ASCII: {
#ifdef ALIEN_USE_LIBXML2
      xmlChar* contenu = xmlNodeGetContent(node.x_id);
      buffer = std::string((char*)contenu);
      xmlFree(contenu);
#endif
    } break;
    case HDF5: {
#ifdef ALIEN_USE_HDF5
      /* Open an existing dataset. */
      hid_t dataset_id = H5Dopen2(node.h_id, "data", H5P_DEFAULT);
      hid_t dataspace_id = H5Dget_space(dataset_id);
      int ndim = H5Sget_simple_extent_ndims(dataspace_id);

      if (ndim != 1) {
        throw FatalErrorException(A_FUNCINFO, "Error while reading HDF5 data set");
      }

      hsize_t size, maxsize;
      herr_t status = H5Sget_simple_extent_dims(dataspace_id, &size, &maxsize);
      if (status) {
        throw FatalErrorException(A_FUNCINFO, "Error while reading HDF5 data set");
      }
      // resize String to size
      buffer.reserve(size + 1);
      buffer.resize(size + 1);
      buffer[size] = '\0';
      status = H5Dread(dataset_id, m_types.nativeType(char()), H5S_ALL, H5S_ALL,
                       H5P_DEFAULT, &buffer[0]);
      if (status) {
        throw FatalErrorException(A_FUNCINFO, "Error while reading HDF5 data set");
      }

      status = H5Sclose(dataspace_id);
      status = H5Dclose(dataset_id);
#endif
    } break;
    default:
      std::cerr << "Unknown output format " << std::endl;
      break;
    }
  }
};

} // namespace Alien
