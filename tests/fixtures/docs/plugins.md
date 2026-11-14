# Plugins

Plugins extend widget with additional build steps. A plugin is a module that exports a function receiving the build context.

## Writing a plugin

The function receives the list of inputs and returns the list of outputs. It must not modify inputs in place, because the build cache relies on them being unchanged.

```ts
export default function upper(ctx) {
  return ctx.inputs.map((file) => ({ ...file, text: file.text.toUpperCase() }));
}
```

## Registering a plugin

Add the module path to the `plugins` list in the configuration file. Plugins run in the listed order, each one receiving the outputs of the previous one.

## Plugin errors

When a plugin throws, widget stops the build, prints the plugin name and the error message, and keeps the previous output folder untouched.

## Plugin options

A plugin entry may be a pair of module path and options object. Widget passes the options object as the second argument to the plugin function.

## Testing plugins

Call the plugin function directly with a hand written context object. Plugins without side effects need no running build to be tested.
