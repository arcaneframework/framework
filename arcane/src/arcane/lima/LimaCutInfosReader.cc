// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LimaCutInfosReader.cc                                       (C) 2000-2026 */
/*                                                                           */
/* Reader for cut information using Lima files.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/lima/LimaCutInfosReader.h"

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/ItemGroup.h"
#include "arcane/core/Item.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/Timer.h"
#include "arcane/core/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//Just to create the .lib under Windows.
class ARCANE_EXPORT LimaTest
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LimaCutInfosReader::
LimaCutInfosReader(IParallelMng* parallel_mng)
: TraceAccessor(parallel_mng->traceMng())
, m_parallel_mng(parallel_mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Reading the correspondences.
 *
 * The correspondences are stored in memory at address \a buf in the form of a sequence of integers in ASCII format separated by spaces.
 * For example: "125 132 256".
 */
static Integer
_readList(Int64ArrayView& int_list, const char* buf)
{
  std::istringstream istr(buf);
  Integer index = 0;
  while (!istr.eof()) {
    Int64 v = 0;
    istr >> v;
    if (istr.eof())
      break;
    int_list[index] = v;
    ++index;
  }
  return index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Constructs the internal mesh structures.
 */
void LimaCutInfosReader::
readItemsUniqueId(Int64ArrayView nodes_id, Int64ArrayView cells_id,
                  const String& dir_name)
{
  _readUniqueIndex(nodes_id, cells_id, dir_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retrieval of unique entity indices.
 *
 * Retrieves the unique number for each entity across all domains.
 * The values are stored in \a nodes_id for the nodes and
 * \a cells_id for the cells. Both arrays must already have been
 * allocated to the correct size.

 * The operation is as follows:
 * \arg processor 0 reads the correspondence file and retrieves the
 * information relevant to it.
 * \arg for each other processor, processor 0 retrieves the number
 * of nodes and cells to read, reads the values from the correspondence file, and transfers them to the processor.
 */
void LimaCutInfosReader::
_readUniqueIndex(Int64ArrayView nodes_id, Int64ArrayView cells_id,
                 const String& dir_name)
{
  Timer time_to_read(m_parallel_mng->timerMng(), "ReadCorrespondance", Timer::TimerReal);
  IParallelMng* pm = m_parallel_mng;

  // If the case is parallel, read the correspondence file.
  bool is_parallel = pm->isParallel();
  Int32 comm_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  // If we are in the case where Arcane is trimmed to one core
  if ((!is_parallel) || (is_parallel && nb_rank == 1)) {
    for (Integer i = 0, n = nodes_id.size(); i < n; ++i)
      nodes_id[i] = i;
    for (Integer i = 0, n = cells_id.size(); i < n; ++i)
      cells_id[i] = i;
    return;
  }

  ScopedPtrT<IXmlDocumentHolder> doc_holder;
  StringBuilder correspondance_filename;
  if (!dir_name.empty()) {
    correspondance_filename += dir_name;
    correspondance_filename += "/";
  }
  correspondance_filename += "Correspondances";

  if (comm_rank == 0) {
    {
      Timer::Sentry sentry(&time_to_read);
      IXmlDocumentHolder* doc = pm->ioMng()->parseXmlFile(correspondance_filename);
      doc_holder = doc;
    }
    if (!doc_holder.get())
      ARCANE_FATAL("Invalid correspondance file '{0}'", correspondance_filename);

    info() << "Time to read (unit: second) 'Correspondances' file: "
           << time_to_read.lastActivationTime();

    XmlNode root_element = doc_holder->documentNode().documentElement();

    // First, the subdomain reads its values.
    _readUniqueIndexFromXml(nodes_id, cells_id, root_element, 0);

    {
      Timer::Sentry sentry(&time_to_read);
      // Then loop over the other subdomains.
      Int64UniqueArray other_nodes_id;
      Int64UniqueArray other_cells_id;
      UniqueArray<Integer> other_sizes(2);
      for (Int32 i = 1; i < nb_rank; ++i) {
        pm->recv(other_sizes, i);
        other_nodes_id.resize(other_sizes[0]);
        other_cells_id.resize(other_sizes[1]);
        _readUniqueIndexFromXml(other_nodes_id, other_cells_id, root_element, i);
        pm->send(other_nodes_id, i);
        pm->send(other_cells_id, i);
      }
    }
    info() << "Time to transfert values: "
           << time_to_read.lastActivationTime();
  }
  else {
    // Receive the values sent by subdomain 0
    Integer mes[2];
    IntegerArrayView mesh_element_size(2, mes);
    mesh_element_size[0] = nodes_id.size();
    mesh_element_size[1] = cells_id.size();
    pm->send(mesh_element_size, 0);
    pm->recv(nodes_id, 0);
    pm->recv(cells_id, 0);
  }
  // Check that all messages are sent and received before destroying
  // the XML document.
  pm->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Reading unique indices from an XML correspondence file.
 * \param nodes_id node indices
 * \param cells_id cell indices
 * \param root_element root element of the XML tree.
 * \param comm_rank subdomain number to read.
 */
void LimaCutInfosReader::
_readUniqueIndexFromXml(Int64ArrayView nodes_id, Int64ArrayView cells_id,
                        XmlNode root_element, Int32 comm_rank)
{
  XmlNode cpu_elem;
  XmlNodeList cpu_list = root_element.children(String("cpu"));

  String ustr_buf(String::fromNumber(comm_rank));

  String us_id("id");
  for (Integer i = 0, s = cpu_list.size(); i < s; ++i) {
    String id_str = cpu_list[i].attrValue(us_id);
    if (id_str.null())
      continue;
    if (id_str == ustr_buf) {
      cpu_elem = cpu_list[i];
      break;
    }
  }
  if (cpu_elem.null())
    ARCANE_FATAL("No element <cpu[@id=\"{0}\"]>", comm_rank);
  XmlNode node_elem = cpu_elem.child("noeuds");
  if (node_elem.null())
    ARCANE_FATAL("No element <noeuds>");
  XmlNode cell_elem = cpu_elem.child("mailles");
  if (cell_elem.null())
    ARCANE_FATAL("No element <mailles>");

  // Node correspondence array
  {
    String ustr_value = node_elem.value();
    Integer nb_read = _readList(nodes_id, ustr_value.localstr());
    Integer expected_size = nodes_id.size();
    if (nb_read != expected_size)
      ARCANE_FATAL("Bad number of nodes rank={0} nb_read={1} expected={2}",
                   comm_rank, nb_read, expected_size);
  }

  // Cell correspondence array
  {
    String ustr_value = cell_elem.value();
    Integer nb_read = _readList(cells_id, ustr_value.localstr());
    Integer expected_size = cells_id.size();
    if (nb_read != expected_size)
      ARCANE_FATAL("Bad number of cells rank={0} nb_read={1} expected={2}",
                   comm_rank, nb_read, expected_size);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
