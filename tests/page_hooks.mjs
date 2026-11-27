const EXPORTS = {
    '/scripts/app.js': ['app', 'ComfyApp'],
    '/scripts/api.js': ['api'],
    '/scripts/ui.js': ['ComfyDialog', '$el'],
    '/ComfyUI-Impact-Pack/impact-pack.js': ['get_wildcards_list', 'get_wildcard_label', 'is_wildcard_label', 'load_wildcard_status'],
};

export async function resolve(specifier, context, next) {
    const parent = context.parentURL && new URL(context.parentURL);
    const page = parent?.searchParams.get('page');
    if (!page)
        return next(specifier, context);

    const module = Object.keys(EXPORTS).find(name => specifier.endsWith(name));
    if (module) {
        const source = `const m = globalThis.inspireTestPages[${page}][${JSON.stringify(module)}];
if (!m) throw new Error(${JSON.stringify(`${module} is not on this page`)});
export const { ${EXPORTS[module].join(', ')} } = m;`;
        return { url: 'data:text/javascript,' + encodeURIComponent(source), shortCircuit: true };
    }

    const url = new URL(specifier, parent);
    url.search = parent.search;
    return { url: url.href, shortCircuit: true };
}
