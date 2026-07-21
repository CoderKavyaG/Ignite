// Declare module for WGSL raw files imported in Vite using the ?raw suffix.
declare module "*.wgsl?raw" {
    const content: string;
    export default content;
}
