import os
import shutil
import bpy
from . import helper, blender_nerf_operator
from pdb import set_trace as st


# global addon script variables
EMPTY_NAME = 'BlenderNeRF Sphere'
CAMERA_NAME = 'BlenderNeRF Camera'

# camera on sphere operator class
class CameraOnSphere(blender_nerf_operator.BlenderNeRF_Operator):
    '''Camera on Sphere Operator'''
    bl_idname = 'object.camera_on_sphere'
    bl_label = 'Camera on Sphere COS'

    def execute(self, context):
        scene = context.scene
        camera = scene.camera

        # reset generator in case of fixed cameras so we will always get the same cameras for each frame
        if scene.fixed_cameras:
            scene.current_camera = 1

        # check if camera is selected : next errors depend on an existing camera
        if camera == None:
            self.report({'ERROR'}, 'Be sure to have a selected camera!')
            return {'FINISHED'}

        # if there is an error, print first error message
        error_messages = self.asserts(scene, method='COS')
        if len(error_messages) > 0:
           self.report({'ERROR'}, error_messages[0])
           return {'FINISHED'}

        is_animation = scene.animate and self.get_total_keyframes_in_scene(scene) > 0

        # clean directory name (unsupported characters replaced) and output path
        output_dir = bpy.path.clean_name(scene.cos_dataset_name)
        output_path = os.path.join(scene.save_path, output_dir)
        os.makedirs(output_path, exist_ok=True)

        if scene.logs: self.save_log_file(scene, output_path, method='COS')
        if scene.splats: self.save_splats_ply(scene, output_path)

        # initial property might have changed since set_init_props update
        scene.init_output_path = scene.render.filepath

        # other intial properties
        scene.init_sphere_exists = scene.show_sphere
        scene.init_camera_exists = scene.show_camera
        scene.init_frame_end = scene.frame_end
        scene.init_active_camera = camera
        
        if scene.test_data:
            if (scene.render_test_frames or scene.debug) and not scene.show_camera: 
                scene.show_camera = True
            
            if scene.render_test_frames or scene.debug:
                # train camera on sphere
                sphere_camera = scene.objects[CAMERA_NAME]
                output_data = self.get_camera_intrinsics(scene, sphere_camera)
                scene.camera = sphere_camera
            else:
                output_data = self.get_camera_intrinsics(scene, camera)

            camera = scene.camera
            
            # testing transforms
            output_data['frames'], test_locations = self.get_camera_extrinsics(scene, camera, mode='TEST', 
                                                            method='COS', is_animation=is_animation)
            if is_animation:
                self.append_to_json_file(output_path, 'transforms_test.json', output_data)
            else:
                self.save_json(output_path, 'transforms_test.json', output_data)
            
            if scene.render_test_frames:
                output_test = os.path.join(output_path, 'test')
                os.makedirs(output_test, exist_ok=True)
                scene.rendering = (False, False, True)
                
                for i in range(scene.cos_nb_test_frames):
                    camera.location = test_locations[i]
                    context.view_layer.update()
                    if not is_animation:
                        scene.render.filepath = os.path.join(output_test, '') # testing frames path
                    else:
                        im_name = 'frame_{}_view_{}.png'.format(str(scene.frame_current).zfill(4), str(i+1).zfill(4))
                        scene.render.filepath = os.path.join(output_test, im_name) # testing frames path
                    
                    bpy.ops.render.render('INVOKE_DEFAULT', animation=(not is_animation), write_still=True) # render scene
                
                # if not is_animation:
                #     for i in range(scene.cos_nb_test_frames):
                #         # scene.frame_end = scene.frame_start + scene.cos_nb_test_frames - 1 # update end frame
                #         camera.location = test_locations[i]
                #         context.view_layer.update()
                #         scene.render.filepath = os.path.join(output_test, '') # training frames path
                #         bpy.ops.render.render('INVOKE_DEFAULT', animation=True, write_still=True) # render scene
                # else:
                #     for i in range(scene.cos_nb_test_frames):
                #         camera.location = test_locations[i]
                #         context.view_layer.update()
                #         im_name = 'frame_{}_view_{}.png'.format(str(scene.frame_current).zfill(4), str(i+1).zfill(4))
                #         scene.render.filepath = os.path.join(output_test, im_name)
                #         bpy.ops.render.render('INVOKE_DEFAULT', animation=False, write_still=True) # render scene
        
        if scene.train_data:
            if not scene.show_camera: scene.show_camera = True

            # train camera on sphere
            sphere_camera = scene.objects[CAMERA_NAME]
            sphere_output_data = self.get_camera_intrinsics(scene, sphere_camera)
            scene.camera = sphere_camera

            # training transforms
            sphere_output_data['frames'], train_locations = self.get_camera_extrinsics(scene, sphere_camera, mode='TRAIN', 
                                                                      method='COS', is_animation=is_animation)
            if is_animation:
                self.append_to_json_file(output_path, 'transforms_train.json', sphere_output_data)
            else:
                self.save_json(output_path, 'transforms_train.json', sphere_output_data)

            # rendering
            if scene.render_frames:
                output_train = os.path.join(output_path, 'train')
                os.makedirs(output_train, exist_ok=True)
                scene.rendering = (False, False, True)
                
                
                for i in range(scene.cos_nb_frames):
                    sphere_camera.location = train_locations[i]
                    context.view_layer.update()
                    if not is_animation:
                        scene.render.filepath = os.path.join(output_train, '') # training frames path
                    else:
                        im_name = 'frame_{}_view_{}.png'.format(str(scene.frame_current).zfill(4), str(i+1).zfill(4))
                        scene.render.filepath = os.path.join(output_train, im_name) # training frames path
                    
                    bpy.ops.render.render('INVOKE_DEFAULT', animation=(not is_animation), write_still=True) # render scene

                # if not is_animation:
                #     for i in range(scene.cos_nb_frames):
                #         sphere_camera.location = train_locations[i]
                #         context.view_layer.update()
                #         # scene.frame_end = scene.frame_start + scene.cos_nb_frames - 1 # update end frame
                #         scene.render.filepath = os.path.join(output_train, '') # training frames path
                #         bpy.ops.render.render('INVOKE_DEFAULT', animation=True, write_still=True) # render scene
                # else:
                #     for i in range(scene.cos_nb_frames):
                #         sphere_camera.location = train_locations[i]
                #         context.view_layer.update()
                #         im_name = 'frame_{}_view_{}.png'.format(str(scene.frame_current).zfill(4), str(i+1).zfill(4))
                #         scene.render.filepath = os.path.join(output_train, im_name) # training frames path
                #         bpy.ops.render.render('INVOKE_DEFAULT', animation=False, write_still=True) # render scene
                

        # if frames are rendered, the below code is executed by the handler function
        # if not any(scene.rendering):
        #     # reset camera settings
        #     if not scene.init_camera_exists: helper.delete_camera(scene, CAMERA_NAME)
        #     if not scene.init_sphere_exists:
        #         objects = bpy.data.objects
        #         objects.remove(objects[EMPTY_NAME], do_unlink=True)
        #         scene.show_sphere = False
        #         scene.sphere_exists = False

        #     scene.camera = scene.init_active_camera

        #     # compress dataset and remove folder (only keep zip)
        #     shutil.make_archive(output_path, 'zip', output_path) # output filename = output_path
        #     shutil.rmtree(output_path)

        return {'FINISHED'}