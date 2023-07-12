#include "alien/arcane_tools/accessors/ItemVectorAccessor.h"

/*---------------------------------------------------------------------------*/

#define ALIEN_ACCESSOR_ITEMVECTORACCESSOR_CC

// Si on est en debug ou qu'on ne souhaite pas l'inlining, VectorAccessorT est inclus ici
#ifdef ALIEN_INCLUDE_TEMPLATE_IN_CC
#include "alien/arcane_tools/Accessors/ItemVectorAccessorT.h"
#endif /* SHOULD_BE_INCLUDE_IN_CC */

#undef ALIEN_ACCESSOR_ITEMVECTORACCESSOR_CC

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template class ALIEN_ARCANE_TOOLS_EXPORT ItemVectorAccessorT<double>;

#ifdef WIN32
  template class ALIEN_ARCANE_TOOLS_EXPORT ItemVectorAccessorT<double>::VectorElement;
  template void ALIEN_ARCANE_TOOLS_EXPORT ItemVectorAccessorT<double>::VectorElement::
  operator=(const double&);
#endif

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
