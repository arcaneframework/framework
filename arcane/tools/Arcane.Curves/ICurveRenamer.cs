using System;
namespace Arcane.Curves
{
  public interface ICurveRenamer
  {
    /// <summary>
    /// Retourne le nouveau nom de la courbe.
    /// </summary>
    /// <returns>The rename.</returns>
    /// <param name="original_name">Nom d'origine</param>
    string Rename(string original_name);
  }
}
