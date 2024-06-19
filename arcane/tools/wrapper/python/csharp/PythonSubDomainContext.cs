using Arcane;

namespace Arcane.Python
{
  public class SubDomainContext
  {
    public SubDomainContext(ISubDomain sd)
    {
      m_sub_domain = sd;
    }
    public string Name() { return "SubDomain"; }
    readonly ISubDomain m_sub_domain;
  }
}
