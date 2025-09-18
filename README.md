# Registro UCI · IPS (Frontend en GitHub Pages)

Este sitio embebe el formulario publicado como **Google Apps Script Web App**.

## Despliegue

1. Implementar el Web App en Google Apps Script (obtener URL `/exec`).
2. Editar `index.html` y reemplazar `GAS_WEB_APP_URL` por la URL de tu despliegue.
3. Activar GitHub Pages en Settings → Pages (Branch: `main`, Folder: `/root`).
4. Esperar la publicación (1–3 min). Acceder a: [https://apoyomedicoips.github.io/uciips/]

## Notas
- El formulario y la lógica residen en Google Apps Script.
- Este repo sirve como **portón público** (landing + iframe).
- Para cambiar catálogos de listas, ver hoja `diccionario` en el Spreadsheet.
