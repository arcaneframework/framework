/*---------------------------------------------------------------------------*/
/* ArcaneCodeService.cc                                        (C) 2000-2012 */
/*                                                                           */
/* Service de code générique Arcane.                                         */
/* Ce service est une recopie locale de celui de tests pour le driver        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/CodeService.h"
#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/ISession.h"
#include "arcane/ISubDomain.h"
#include "arcane/IParallelMng.h"
#include "arcane/Service.h"

#include "arcane/impl/TimeLoopReader.h"

#include "arcane/std/ArcaneSession.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneCodeService : public CodeService
{
 public:
  ArcaneCodeService(const ServiceBuildInfo& sbi);

  virtual ~ArcaneCodeService();

 public:
  virtual bool parseArgs(StringList& args);

  virtual ISession* createSession();

  virtual void initCase(ISubDomain* sub_domain, bool is_continue);

 public:
  void build() {}

 protected:
  virtual void _preInitializeSubDomain(ISubDomain* sd);

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneCodeService::ArcaneCodeService(const ServiceBuildInfo& sbi)
: CodeService(sbi)
{
  _addExtension(String("arc"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneCodeService::~ArcaneCodeService()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
ArcaneCodeService::_preInitializeSubDomain(ISubDomain*)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
ArcaneCodeService::initCase(ISubDomain* sub_domain, bool is_continue)
{
  {
    TimeLoopReader stl(_application());
    stl.readTimeLoops();
    stl.registerTimeLoops(sub_domain);
    stl.setUsedTimeLoop(sub_domain);
  }
  CodeService::initCase(sub_domain, is_continue);
  if (sub_domain->parallelMng()->isMasterIO())
    sub_domain->session()->writeExecInfoFile();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISession*
ArcaneCodeService::createSession()
{
  ArcaneSession* session = new ArcaneSession(_application());
  session->build();
  _application()->addSession(session);
  return session;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool
ArcaneCodeService::parseArgs(StringList& args)
{
  ARCANE_UNUSED(args);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(ArcaneCodeService, ICodeService, ArcaneCode);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
