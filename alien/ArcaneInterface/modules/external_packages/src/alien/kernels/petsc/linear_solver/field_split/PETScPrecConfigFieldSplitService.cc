/* Author : gratienj
 * Preconditioner created by combining separate preconditioners for individual
 * fields or groups of fields. See the users manual section "Solving Block Matrices"
 * for more details in PETSc 3.3 documentation :
 * http://www.mcs.anl.gov/petsc/petsc-current/docs/manual.pdf
 */

#include <alien/kernels/petsc/linear_solver/field_split/PETScPrecConfigFieldSplitService.h>

#include <ALIEN/axl/PETScPrecConfigFieldSplit_StrongOptions.h>

#include <alien/core/utils/Partition.h>

#include <arccore/message_passing/IMessagePassingMng.h>

#include <map>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
PETScPrecConfigFieldSplitService::PETScPrecConfigFieldSplitService(
    const Arcane::ServiceBuildInfo& sbi)
: ArcanePETScPrecConfigFieldSplitObject(sbi)
, PETScConfig(sbi.subDomain()->parallelMng()->isParallel())
{
  m_default_block_tag = "default";
}
#endif
PETScPrecConfigFieldSplitService::PETScPrecConfigFieldSplitService(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsPETScPrecConfigFieldSplit> options)
: ArcanePETScPrecConfigFieldSplitObject(options)
, PETScConfig(parallel_mng->commSize() > 1)
{
  m_default_block_tag = "default";
}

//! Initialisation
void
PETScPrecConfigFieldSplitService::configure(
    PC& pc, const ISpace& space, const MatrixDistribution& distribution)
{
  alien_debug([&] { cout() << "configure PETSc FlieldSplit preconditioner"; });
  /*const Integer blocks_size = options()->block().size();
    traceMng()->error() << "blocks_size 1 "<< blocks_size;
    IFieldSplitType* field_split_solver2 = options()->type();
    if(field_split_solver2 == NULL)
          traceMng()->fatal() << "field split solver null";*/
  /*if(options()->verbose()){
      traceMng()->error() << "verbose";
    }
    else{
      traceMng()->error() << "no verbose";
    }*/
  /*Arcane::ConstArrayView<IOptionsPETScPrecConfigFieldSplit::IFieldSolver*> block =
    options()->block(); const Integer blocks_size2 = block.size();
    traceMng()->error() << "blocks_size 2 "<< blocks_size2;*/
  checkError("Set preconditioner", PCSetType(pc, PCFIELDSPLIT));

  checkError("Build FieldSplit IndexSet", this->initializeFields(space, distribution));

  const Arccore::Integer nbFields = m_field_petsc_indices.size();
  ALIEN_ASSERT(
      (not m_field_petsc_indices.empty()), ("Unexpected empty PETSc IS for FieldSplit"));

  for (Arccore::Integer i = 0; i < nbFields; ++i) {
#if ((PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR < 3) || (PETSC_VERSION_MAJOR < 3))
    checkError("Set PetscIS", PCFieldSplitSetIS(pc, m_field_petsc_indices[i]));
#else /* PETSC_VERSION */
    checkError("Set PetscIS",
        PCFieldSplitSetIS(pc, m_field_tags[i].localstr(), m_field_petsc_indices[i]));
#endif /* PETSC_VERSION */
  }

  IFieldSplitType* field_split_solver = options()->type();

  if (field_split_solver == NULL)
    alien_fatal([&] { cout() << "field split solver null"; });

  // Configure type of FieldSplit decomposition
  checkError("Set FieldSplit type", field_split_solver->configure(pc, nbFields));

  PCSetUp(pc);

  Arccore::Integer nbf;
  KSP* subksp;
  checkError("Get FieldSplit Sub KSP", PCFieldSplitGetSubKSP(pc, &nbf, &subksp));

  if (nbf != nbFields)
    alien_fatal([&] {
      cout() << "Inconsistent number of split : user=" << nbFields << " PETSc=" << nbf;
    });

  // a revoir pas naturel
  for (Arccore::Integer i = 0; i < nbFields; ++i) {
    if (m_field_tags[i] == m_default_block_tag) {
      IPETScKSP* sub_solver = options()->defaultBlock()[0]->solver();
      sub_solver->configure(subksp[i], space, distribution);
    } else {
      // pas bon verifier manque indirection
      IPETScKSP* sub_solver = options()->block()[i]->solver();
      sub_solver->configure(subksp[i], space, distribution);
    }
  }
}

Arccore::Integer
PETScPrecConfigFieldSplitService::initializeFields(
    const ISpace& space, const MatrixDistribution& distribution)
{
  const Arccore::String block_tag =
      options()
          ->blockTag(); // m_parent->getParams<std::string>()["fieldsplit-block-tag"];

  // Verification des doublons de tags
  std::set<Arccore::String> tag_set;
  // const Integer block_size = m_parent->getParams<Integer>()["fieldsplit-block-size"];
  const Arccore::Integer blocks_size = options()->block().size();
  // traceMng()->error() << "blocks_size "<< blocks_size;
  m_field_tags.clear();
  for (Arccore::Integer i = 0; i < blocks_size; ++i) {
    Arccore::String tag = options()->block()[i]->tag();
    // traceMng()->error() << "tag "<< tag;
    if (tag == m_default_block_tag)
      alien_fatal([&] { cout() << "block-tag 'default' is a reserved keyword"; });
    std::pair<std::set<Arccore::String>::const_iterator, bool> inserter =
        tag_set.insert(tag);
    if (inserter.second) {
      // traceMng()->error() << "add(m_field_tags,tag); "<< tag;
      m_field_tags.add(tag);
    } else
      alien_fatal([&] { cout() << "Duplicate block-tag found : " << tag; });
  }

  // Partiton des indices
  Partition partition(space, distribution);
  partition.create(m_field_tags);

  bool has_untagged_part = partition.hasUntaggedPart();
  //
  bool has_default_block = (options()->defaultBlock().size() > 0);

  // A t'on un defaut pour les partitions non taggés?
  if (has_untagged_part && (not has_default_block)) {
    alien_fatal(
        [&] { cout() << "Partition has untagged part and no default block is allowed"; });
    return 1;
  }

  Arccore::Integer nbPart = partition.nbTaggedParts();

  m_field_petsc_indices.clear();
  m_field_petsc_indices.resize(nbPart + has_untagged_part);

  Arccore::Integer verbosity = options()->verbose();

  Arccore::UniqueArray<Arccore::Integer> current_field_indices;
  Arccore::Integer nerror = 0;
  // Création de l'index set pour PETSc
  // Incrémentation de nerror si partition vide
  // Utilisation de current_field_indices comme buffer
  auto createIS = [&](Arccore::String tag,
      const Arccore::UniqueArray<Arccore::Integer>& indices, IS& petsc_is) {
    if (verbosity)
      alien_info(
          [&] { cout() << "Tag '" << tag << "' has " << indices.size() << " indices"; });

    if (indices.size() == 0) {
      // Si partition vide, erreur
      alien_fatal([&] { cout() << "No entry found for block-tag '" << tag << "'"; });
      ++nerror;
    } else {
      // copie et tri => peut être évité si besoin
      current_field_indices.resize(0);
      current_field_indices.copy(indices);
      std::sort(current_field_indices.begin(),
          current_field_indices.end()); // PETSc requires index ordering

// Creation de l'index set pour PETSc
#if ((PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR < 3) || (PETSC_VERSION_MAJOR < 3))
      checkError("Create IndexSet",
          ISCreateGeneral(PETSC_COMM_WORLD, current_field_indices.size(),
              unguardedBasePointer(current_field_indices), &petsc_is));
#else /* PETSC_VERSION */
#ifndef WIN32
#warning "TODO OPTIM: using other copy mode may be more efficient"
#endif
      checkError("Create IndexSet",
          ISCreateGeneral(PETSC_COMM_WORLD, current_field_indices.size(),
              current_field_indices.unguardedBasePointer(), PETSC_COPY_VALUES,
              &petsc_is));
#endif /* PETSC_VERSION */
    }
  };

  // Ajout des partitions taggées
  for (Arccore::Integer i = 0; i < partition.nbTaggedParts(); ++i) {
    createIS(m_field_tags[i], partition.taggedPart(i), m_field_petsc_indices[i]);
  }

  // Partition non taggée
  if (has_untagged_part) {
    m_field_tags.add(m_default_block_tag);
    createIS(
        m_field_tags[nbPart], partition.untaggedPart(), m_field_petsc_indices[nbPart]);
  }

  return nerror;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// using namespace Arcane;
ARCANE_REGISTER_SERVICE_PETSCPRECCONFIGFIELDSPLIT(
    FieldSplit, PETScPrecConfigFieldSplitService);

} // namespace Alien

REGISTER_STRONG_OPTIONS_PETSCPRECCONFIGFIELDSPLIT();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
